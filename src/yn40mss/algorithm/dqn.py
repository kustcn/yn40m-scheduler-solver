import logging
import os
import pickle
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from astropy.time import Time

from ..core.utils import cal_avail_times, get_obs_site, iers_init, load_targets, minutes
from .dqn_env import TelescopeEnv
from .dqn_network import EnhancedAttentionNetwork


# ====================== 智能体 ======================
class PrioritizedReplayBuffer:
    """优先经验回放缓冲区: 按 p^alpha 概率采样, 带重要性采样权重校正"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.tree_idx = 0
        self.size = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.buffer = [None] * capacity

    def add(self, experience: Tuple, priority: Optional[float] = None) -> None:
        if priority is None:
            priority = self.max_priority
        idx = self.tree_idx % self.capacity
        self.buffer[idx] = experience
        self.priorities[idx] = priority ** self.alpha
        self.tree_idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List, List[int], np.ndarray]:
        if self.size < batch_size:
            available_indices = list(range(self.size))
            indices = random.choices(available_indices, k=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
            weights = (self.size * probabilities[indices]) ** (-self.beta)
            weights = weights / np.max(weights)
        self.beta = min(1.0, self.beta + self.beta_increment)
        experiences = [self.buffer[idx] for idx in indices]
        return experiences, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.size:
                self.priorities[idx] = priority ** self.alpha
                self.max_priority = max(self.max_priority, priority)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and torch.backends.mps.is_available():
        try:
            t = torch.zeros(2, device='mps')
            _ = t @ t
            return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')


class DRLAgent:
    """决斗注意力 DQN 智能体(Double Q + PER + 软更新), 与 v1 一致"""

    def __init__(self, state_dim: int, action_dim: int,
                 obs_feat_dim: int = TelescopeEnv.OBS_FEAT_DIM,
                 target_feat_dim: int = TelescopeEnv.TARGET_FEAT_DIM):
        self.device = _select_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_feat_dim = obs_feat_dim
        self.target_feat_dim = target_feat_dim
        self.use_mixed_precision = self.device.type == 'cuda'

        net_kwargs = dict(input_dim=state_dim, action_dim=action_dim,
                          embedding_dim=128, num_heads=8,
                          obs_feat_dim=obs_feat_dim, target_feat_dim=target_feat_dim)
        self.policy_net = EnhancedAttentionNetwork(**net_kwargs).to(self.device)
        self.target_net = EnhancedAttentionNetwork(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.memory = PrioritizedReplayBuffer(capacity=100000)

        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.eps = 1e-6
        self.scaler = torch.amp.GradScaler('cuda') if self.use_mixed_precision else None

    def act(self, state: np.ndarray, env: TelescopeEnv, epsilon: float = 0.0) -> int:
        valid_actions = env._get_valid_actions()
        if not valid_actions:
            return -1
        if random.random() < epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)
            mask = torch.full((self.action_dim,), -float('inf'), device=self.device)
            mask[valid_actions] = 0.0
            return (q_values + mask).argmax().item()

    def store_transition(self, state, action, reward, next_state, done) -> None:
        if action is None or action < 0:
            return
        self.memory.add((torch.FloatTensor(state), int(action), float(reward),
                         torch.FloatTensor(next_state), bool(done)))

    def update(self) -> Optional[float]:
        if self.memory.size < self.batch_size:
            return None
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        def _loss():
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
            return (elementwise_loss * weights.unsqueeze(1)).mean(), \
                torch.abs(current_q - target_q).detach().cpu().numpy()

        if self.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                loss, td_errors = _loss()
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, td_errors = _loss()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.squeeze() + self.eps)
        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data
                                    + (1 - self.tau) * target_param.data)
        return loss.item()


def initialize(cfg):
    iers_init(cfg)


def save_agent(agent: DRLAgent, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'state_dim': agent.state_dim,
        'action_dim': agent.action_dim,
        'obs_feat_dim': agent.obs_feat_dim,
        'target_feat_dim': agent.target_feat_dim,
        'optimizer': agent.optimizer.state_dict(),
    }, filepath)
    logging.info(f"Agent saved to {filepath}")


def load_agent(filepath: str) -> DRLAgent:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    agent = DRLAgent(
        state_dim=checkpoint['state_dim'],
        action_dim=checkpoint['action_dim'],
        obs_feat_dim=checkpoint.get('obs_feat_dim', TelescopeEnv.OBS_FEAT_DIM),
        target_feat_dim=checkpoint.get('target_feat_dim', TelescopeEnv.TARGET_FEAT_DIM),
    )
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    if 'optimizer' in checkpoint:
        try:
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            logging.warning(f"optimizer state not loaded: {e}")
    logging.info(f"Agent loaded from {filepath}")
    return agent


# ====================== 窗口缓存(与 v1 相同) ======================
def _load_or_compute_windows(targets, observer, time_range, cfg) -> dict:
    cache_path = 'target_available_times.pkl'  # 当前工作目录, 便于复用与清理
    start_unix, end_unix = float(time_range[0].unix), float(time_range[1].unix)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get('version') == 2 \
                    and abs(cached['start_unix'] - start_unix) < 60 \
                    and abs(cached['end_unix'] - end_unix) < 60:
                logging.info("Loading cached available times (time range matched)")
                return cached['windows']
        except Exception as e:
            logging.warning(f"Failed to load window cache: {e}")

    logging.info("Computing available times from scratch")
    avail_times, _ = cal_avail_times(targets, observer, time_range, cfg)
    windows = {name: [(float(s.unix) - start_unix, float(e.unix) - start_unix)
                      for s, e in wins]
               for name, wins in avail_times.items()}
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({'version': 2, 'start_unix': start_unix, 'end_unix': end_unix,
                         'windows': windows}, f)
        logging.info(f"Window cache written to {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to write window cache: {e}")
    return windows


# ====================== 调度质量评价 ======================
def _key(entries, fg_set=None):
    """字典序 key: 不同目标出现数(全部目标, 含 fill-gap 源的必排) -> 总观测 -> -总转向

    fill-gap 源同样是必排目标: 其唯一观测被 destroy 移除后 distinct 立即 -1,
    key/score 同步下降, 搜索必然拒绝该候选 —— 保证任何源的必排身份不被牺牲。
    (fg_set 参数保留以兼容旧调用, 语义上不再参与评价。)
    """
    if not entries:
        return (0, 0.0, 0.0)
    distinct = len({e['target_idx'] for e in entries})
    total_obs = sum(e['duration'] for e in entries)
    total_slew = sum(e.get('slew_time', 0.0) for e in entries)
    return (distinct, total_obs, -total_slew)


def _score(entries, fg_set=None) -> float:
    """阈值接受用的标量化分数: 1000*不同目标数 + 总观测 - 总转向

    每个目标(必排, 含 fill-gap 源)权重 1000, 远超单条观测(48 min)与总
    转向之和, 保证阈值接受的小幅劣化永远不会牺牲任何一个目标的必排出现。
    """
    mand, total_obs, neg_slew = _key(entries, fg_set)
    return 1000.0 * mand + total_obs + neg_slew


def _validate_entries(env: TelescopeEnv, entries) -> list:
    errors = []
    ordered = sorted(entries, key=lambda e: e['start_time'])
    prev_end_sec, prev_alt, prev_az = 0.0, env.init_alt, env.init_az
    for e in ordered:
        idx = e['target_idx']
        s, ed = e['start_time'] * 60.0, e['end_time'] * 60.0
        target = env.targets[idx]
        if not any(s >= ws - 1e-6 and ed <= we + 1e-6
                   for ws, we in target['time_windows']):
            errors.append(f"target {idx} obs [{s / 60:.1f}, {ed / 60:.1f}]min not inside any window")
        if ed > env.obs_duration + 1e-6:
            errors.append(f"target {idx} exceeds total duration")
        alt, az = env._target_altaz(idx, s)
        need = env._slew_between_sec(prev_alt, prev_az, alt, az)
        if s - prev_end_sec < need - 1.0:
            errors.append(f"target {idx} insufficient slew: "
                          f"gap {s - prev_end_sec:.0f}s < needed {need:.0f}s")
        prev_end_sec, prev_alt, prev_az = ed, *env._target_altaz(idx, ed)
    return errors


def _rollout(agent: Optional[DRLAgent], env: TelescopeEnv, epsilon: float) -> list:
    state = env.reset()
    done = False
    max_steps = 4 * env.n_targets + 200
    steps = 0
    while not done and steps < max_steps:
        if agent is None:
            valid = env._get_valid_actions()
            action = random.choice(valid) if valid else -1
        else:
            action = agent.act(state, env, epsilon)
        state, _, done, _ = env.step(action)
        steps += 1
    return [dict(e) for e in env.schedule]


def _fill_gaps(env: TelescopeEnv, entries: list, fg_indices: List[int],
               max_rounds: int = 40) -> int:
    """多轮把 fill_gap=1 目标(允许重复观测)插入空闲空隙, 返回插入条数"""
    if not fg_indices:
        return 0
    added = 0
    for _ in range(max_rounds):
        round_added = 0
        for idx in fg_indices:
            if env._try_insert(entries, idx):
                round_added += 1
        added += round_added
        if round_added == 0:
            break
    return added


# ====================== v2 核心: 自适应多算子 LNS ======================
class AdaptiveLNS:
    """自适应多算子大邻域搜索

    破坏算子:
      random        随机移除 k 条
      worst_slew    移除转向开销最大的 k 条(压总转向)
      time_related  以随机条目为种子, 按 start_time 邻近移除 k 条
      sky_related   以随机条目为种子, 按观测开始时天区角距邻近移除 k 条

    自适应: 各算子按近期成功率(命中 accepted 解)轮盘选择
    接受准则: 阈值接受 score(cand) >= score(inc) - T, T 随预算线性衰减到 0;
              最终解仍按完整字典序 key 择优, 必排覆盖绝不被牺牲
    """

    OPERATORS = ('random', 'worst_slew', 'time_related', 'sky_related')

    def __init__(self, env: TelescopeEnv, fg_set: Set[int], fg_indices: List[int],
                 fill_in_recreate: bool = True):
        self.env = env
        self.fg_set = fg_set
        self.fg_indices = list(fg_indices)
        # 消融开关: False 时 recreate 不做 fill-gap 重复插入(v1 语义, 仅 post-hoc)
        self.fill_in_recreate = fill_in_recreate

    # ---- 破坏算子 ----
    def _destroy(self, entries: list, op: str, k: int, rng: random.Random) -> list:
        n = len(entries)
        k = min(k, max(1, n - 1))
        if op == 'random':
            victims = set(rng.sample(range(n), k))
        elif op == 'worst_slew':
            # 转向最大的前 2k 条里随机取 k(保留多样性)
            order = sorted(range(n), key=lambda i: -entries[i].get('slew_time', 0.0))
            victims = set(rng.sample(order[:2 * k], k))
        elif op == 'time_related':
            seed = rng.randrange(n)
            order = sorted(range(n), key=lambda i: abs(entries[i]['start_time']
                                                       - entries[seed]['start_time']))
            victims = set(order[:k])
        else:  # sky_related
            seed = rng.randrange(n)
            sa, sz = entries[seed]['alt'], entries[seed]['az']

            def sky_dist(i):
                d_az = abs((entries[i]['az'] - sz + 180.0) % 360.0 - 180.0)
                d_alt = abs(entries[i]['alt'] - sa)
                return d_az + d_alt

            order = sorted(range(n), key=sky_dist)
            victims = set(order[:k])
        return [e for i, e in enumerate(entries) if i not in victims]

    # ---- 重建: 必排重排 + fill-gap 重复插入(v2 整合点) ----
    def _recreate(self, partial: list, rng: random.Random) -> list:
        cand = self.env.fill_remaining([dict(e) for e in partial],
                                       rng=rng, order_noise=rng.choice(
                                           [600.0, 1800.0, 3600.0, 1e9]))
        if self.fill_in_recreate:
            _fill_gaps(self.env, cand, self.fg_indices)
        return cand

    # ---- 主循环 ----
    def run(self, seed_entries: list, time_budget: float, rng: random.Random,
            log_every: int = 0, T0: float = 40.0) -> list:
        env = self.env
        fg = self.fg_set
        incumbent = [dict(e) for e in seed_entries]
        best = [dict(e) for e in seed_entries]
        inc_score = _score(incumbent, fg)
        best_key = _key(best, fg)

        weights = {op: 1.0 for op in self.OPERATORS}
        hits = {op: 0 for op in self.OPERATORS}
        uses = {op: 0 for op in self.OPERATORS}
        segment = 50
        self.final_weights = dict(weights)

        started = time.time()
        it = 0
        while time.time() - started < time_budget:
            elapsed_frac = min((time.time() - started) / time_budget, 1.0)
            T = T0 * (1.0 - elapsed_frac)

            # 轮盘选算子
            total_w = sum(weights.values())
            r = rng.random() * total_w
            op = self.OPERATORS[-1]
            acc = 0.0
            for name in self.OPERATORS:
                acc += weights[name]
                if r <= acc:
                    op = name
                    break
            uses[op] += 1

            k = rng.randint(2, 8)
            partial = self._destroy(incumbent, op, k, rng)
            cand = self._recreate(partial, rng)

            cand_score = _score(cand, fg)
            if cand_score >= inc_score - T:
                incumbent = cand
                inc_score = cand_score
                hits[op] += 1
                cand_key = _key(cand, fg)
                if cand_key > best_key:
                    best = [dict(e) for e in cand]
                    best_key = cand_key
                    if log_every:
                        logging.info(f"[ALNS it {it}|{op}] best: mand {best_key[0]}, "
                                     f"obs {best_key[1]:.1f} min, slew {-best_key[2]:.1f} min")

            it += 1
            # 分段更新算子权重(成功率 EWMA, 下限防饿死)
            if it % segment == 0:
                for name in self.OPERATORS:
                    rate = hits[name] / uses[name] if uses[name] else 0.0
                    weights[name] = max(0.05, 0.7 * weights[name] + 0.3 * (rate + 0.05))
                    hits[name] = 0
                    uses[name] = 0
                if log_every:
                    logging.info(f"[ALNS it {it}] weights="
                                 f"{ {n: round(w, 2) for n, w in weights.items()} }")
        self.iterations = it
        self.final_weights = dict(weights)
        return best


# ====================== 训练与调度入口 ======================
def do_schedule(targets, cfg, start_time, end_time, start_az, start_el,
                is_interrupt=False, model_path: Optional[str] = None,
                save_model: bool = True, test_mode: bool = False,
                priorities=None, elevation_quality_weight: float = 1.0,
                schedule_results_file: str = "schedule_results.json",
                training_metrics_file: Optional[str] = None):
    """v2 调度入口: 训练(同 v1) -> 候选池 -> top-3 种子 ALNS 组合打磨 -> 兜底填闲

    Returns:
        recs: 调度结果列表(按 start_time 排序), 含 fill-gap 重复观测条目
    """
    initialize(cfg)
    site, observer = get_obs_site(cfg)
    named_targets = load_targets(targets, site)
    n_targets = len(named_targets)
    logging.info(f"loaded {n_targets} targets")

    time_range_chk = Time([str(start_time), str(end_time)], scale='utc')
    if minutes(time_range_chk[0], time_range_chk[1]) < sum(t.obs_time for t in named_targets):
        logging.error("Total observation time exceeds scheduling window")
        return []

    time_range = Time([str(start_time), str(end_time)], scale='utc')
    t0_unix = float(time_range[0].unix)
    total_sec = float(time_range[1].unix - time_range[0].unix)

    windows = _load_or_compute_windows(named_targets, observer, time_range, cfg)
    env_targets = [
        {
            'ra': t.sky_coord.ra.degree,
            'dec': t.sky_coord.dec.degree,
            'time_windows': windows[t.name],
            'duration': t.obs_time * 60.0,
            'fill_gap': t.fill_gap,
        }
        for t in named_targets
    ]
    fg_set = set(i for i, t in enumerate(env_targets) if t.get('fill_gap', False))
    fg_indices = sorted(fg_set)
    logging.info(f"mandatory targets: {n_targets - len(fg_set)}, "
                 f"fill-gap eligible: {len(fg_set)} {fg_indices}")

    env = TelescopeEnv(
        site=site,
        targets=env_targets,
        init_altaz=(float(start_el), float(start_az)),
        start_time=t0_unix,
        obs_duration=total_sec,
        elevation_quality_weight=elevation_quality_weight,
    )

    # ---- 训练(与 v1 相同的多尝试 D3QN) ----
    agent = None
    if model_path and os.path.exists(model_path):
        try:
            loaded = load_agent(model_path)
            if loaded.state_dim == env.state_dim and loaded.action_dim == env.action_dim:
                agent = loaded
                agent.device = _select_device()
                agent.policy_net.to(agent.device)
                agent.target_net.to(agent.device)
            else:
                logging.warning("Model dims mismatch, training from scratch")
        except Exception as e:
            logging.warning(f"Failed to load model: {e}, training from scratch")
    if agent is None:
        agent = DRLAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    logging.info(f"Using device: {agent.device}")

    attempts = 2 if test_mode else 3
    episodes_per_attempt = 30 if test_mode else 150
    epsilon_start, epsilon_end = 0.9, 0.05
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / episodes_per_attempt)

    best_entries, best_key = [], (-1, -1.0, 0.0)
    model_save_dir = os.path.join(os.getcwd(), 'models')
    metrics = []

    for attempt in range(attempts):
        epsilon = epsilon_start if attempt == 0 else max(epsilon_start * 0.5, epsilon_end)
        logging.info(f"=== Attempt {attempt + 1}/{attempts} start (epsilon={epsilon:.3f}) ===")
        for ep in range(episodes_per_attempt):
            if callable(is_interrupt) and is_interrupt():
                break
            state = env.reset()
            total_reward, done, steps = 0.0, False, 0
            episode_losses = []
            max_steps = 4 * n_targets + 200
            while not done and steps < max_steps:
                action = agent.act(state, env, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                steps += 1
                if steps % 4 == 0:
                    loss = agent.update()
                    if loss is not None:
                        episode_losses.append(loss)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            entries = [dict(e) for e in env.schedule]
            key = _key(entries, fg_set)
            if key > best_key:
                best_entries, best_key = entries, key
                if save_model:
                    save_agent(agent, os.path.join(model_save_dir, 'best_model.pt'))
            metrics.append({
                'attempt': attempt + 1,
                'episode': attempt * episodes_per_attempt + ep,
                'reward': round(total_reward, 2),
                'records': len(entries),
                'mandatory_cov': key[0],
                'total_obs_min': round(key[1], 1),
                'total_slew_min': round(-key[2], 2),
                'loss': round(float(np.mean(episode_losses)), 5) if episode_losses else None,
                'epsilon': round(epsilon, 4),
            })
            if ep % 20 == 0 or ep == episodes_per_attempt - 1:
                avg_loss = float(np.mean(episode_losses)) if episode_losses else float('nan')
                logging.info(f"Episode {attempt * episodes_per_attempt + ep} | "
                             f"Reward: {total_reward:.2f} | Loss: {avg_loss:.4f} | "
                             f"Epsilon: {epsilon:.3f} | Records: {len(env.schedule)}")

    try:
        import json
        metrics_path = training_metrics_file or os.path.join(os.getcwd(), 'dqn_training_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({'variant': 'v2', 'test_mode': test_mode,
                       'attempts': attempts,
                       'episodes_per_attempt': episodes_per_attempt,
                       'records': metrics}, f, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Failed to save training metrics: {e}")

    if save_model:
        save_agent(agent, os.path.join(model_save_dir, 'final_model.pt'))

    # ---- 候选池 ----
    candidates = []
    if best_entries:
        candidates.append(best_entries)
    rollout_epsilons = [0.0] + [0.05] * (2 if test_mode else 8)
    for eps in rollout_epsilons:
        candidates.append(_rollout(agent, env, eps))
    constructive = env.fill_remaining([])
    _fill_gaps(env, constructive, fg_indices)
    candidates.append(constructive)

    scored = sorted(candidates, key=lambda c: _key(c, fg_set), reverse=True)
    seeds = scored[:3]
    for c in seeds:
        k = _key(c, fg_set)
        logging.info(f"seed: mand {k[0]}, obs {k[1]:.1f} min, slew {-k[2]:.1f} min, "
                     f"records {len(c)}")

    # ---- v2: top-3 种子 ALNS 组合打磨 ----
    lns_budget = 15.0 if test_mode else 60.0
    rng = random.Random()
    alns = AdaptiveLNS(env, fg_set, fg_indices)

    phase1 = lns_budget * 0.4
    phase1_results = []
    t0 = time.time()
    for si, seed in enumerate(seeds):
        res = alns.run(seed, phase1 / len(seeds), random.Random(1000 + si), log_every=0)
        phase1_results.append(res)
        k = _key(res, fg_set)
        logging.info(f"ALNS phase1 seed{si}: mand {k[0]}, obs {k[1]:.1f} min, "
                     f"slew {-k[2]:.1f} min, records {len(res)}")

    champion = max(phase1_results, key=lambda c: _key(c, fg_set))
    phase2 = lns_budget * 0.6
    final_entries = alns.run(champion, phase2, random.Random(7), log_every=0)
    logging.info(f"ALNS total: {time.time() - t0:.1f}s "
                 f"(phase1 {phase1:.0f}s + phase2 {phase2:.0f}s)")

    # ---- 兜底填闲(多数情况下 ALNS 已填满) ----
    extra = _fill_gaps(env, final_entries, fg_indices)
    if extra:
        logging.info(f"post-hoc fill-gap added {extra} more observations")
    final_entries.sort(key=lambda e: e['start_time'])

    final_key = _key(final_entries, fg_set)
    mand_cov, total_obs, neg_slew = final_key
    logging.info(f"[v2 final] distinct targets {mand_cov}/{n_targets}, "
                 f"records {len(final_entries)}, obs {total_obs:.1f} min, "
                 f"slew {-neg_slew:.1f} min")

    errors = _validate_entries(env, final_entries)
    if errors:
        logging.warning(f"Schedule validation issues ({len(errors)}): {errors[:5]}")
    else:
        logging.info("Schedule validation passed: windows/slew/overlap all feasible")

    # ---- 输出 ----
    start_datetime = datetime.strptime(str(start_time), "%Y-%m-%d %H:%M:%S")
    recs = []
    for e in final_entries:
        recs.append({
            'target_idx': e['target_idx'],
            'target': named_targets[e['target_idx']].name,
            'fill_gap': bool(env_targets[e['target_idx']].get('fill_gap', False)),
            'slew_time': round(e['slew_time'], 2),
            'start_time': round(e['start_time'], 2),
            'end_time': round(e['end_time'], 2),
            'duration': round(e['duration'], 2),
            'elevation': round(e.get('el_mean', 0.0), 2),
            'start_time_str': (start_datetime + timedelta(minutes=e['start_time'])).strftime("%Y-%m-%d %H:%M:%S"),
            'end_time_str': (start_datetime + timedelta(minutes=e['end_time'])).strftime("%Y-%m-%d %H:%M:%S"),
        })

    try:
        import json
        quality = env.schedule_quality(final_entries)
        result_path = schedule_results_file
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'variant': 'v2',
                'start_time': str(start_time),
                'end_time': str(end_time),
                'total_targets': n_targets,
                'mandatory_targets': n_targets - len(fg_set),
                'mandatory_scheduled': mand_cov,
                'scheduled': len(recs),
                'records': len(recs),
                'unique_targets': len({r['target'] for r in recs}),
                'fill_gap_extra': len(recs) - len({r['target'] for r in recs}),
                'weighted_coverage': round(quality['weighted_coverage'], 4),
                'total_obs_minutes': round(total_obs, 1),
                'total_slew_minutes': round(-neg_slew, 2),
                'mean_elevation_deg': round(quality['mean_elevation'], 2),
                'records': recs,
            }, f, ensure_ascii=False, indent=2)
        logging.info(f"Schedule results saved to {result_path}")
    except Exception as e:
        logging.warning(f"Failed to save schedule results: {e}")

    return recs
