import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import time
import pickle
import logging
from astropy.time import Time
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

from ..core.utils import iers_init, get_obs_site, cal_avail_times, load_targets, minutes
from .dqn_env import TelescopeEnv
from .dqn_network import EnhancedAttentionNetwork

# ====================== 智能体 ======================
# 优先经验回放的存储结构
class PrioritizedReplayBuffer:
    """优先经验回放缓冲区

    按优先级 p^alpha 概率采样,带重要性采样权重校正
    """
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment  # beta的增量
        self.max_priority = 1.0  # 初始最大优先级
        self.tree_idx = 0   # 当前树索引
        self.size = 0       # 当前缓冲区大小

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
        """基于优先级采样,返回 (经验列表, 索引列表, 重要性权重)"""
        if self.size < batch_size:
            available_indices = list(range(self.size))
            indices = random.choices(available_indices, k=batch_size)
            probabilities = None
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        if probabilities is None:
            weights = np.ones(batch_size, dtype=np.float32)
        else:
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
    """cuda > mps > cpu 自动选择"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and torch.backends.mps.is_available():
        try:  # MPS 可用性需实际验证
            t = torch.zeros(2, device='mps')
            _ = t @ t
            return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')


class DRLAgent:
    """深度强化学习智能体

    使用优先经验回放 + Double DQN + 目标网络软更新的 DQN 智能体
    """
    def __init__(self, state_dim: int, action_dim: int,
                 obs_feat_dim: int = TelescopeEnv.OBS_FEAT_DIM,
                 target_feat_dim: int = TelescopeEnv.TARGET_FEAT_DIM):
        self.device = _select_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_feat_dim = obs_feat_dim
        self.target_feat_dim = target_feat_dim
        # 混合精度仅在 CUDA 上启用
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
        """ε-贪婪 + 有效动作掩码选择动作; 无有效动作时返回 -1(等待/推进时间)"""
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

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        if action is None or action < 0:
            return  # 等待/推进动作不属于 Q 动作空间, 不存储
        experience = (
            torch.FloatTensor(state),
            int(action),
            float(reward),
            torch.FloatTensor(next_state),
            bool(done)
        )
        self.memory.add(experience)

    def update(self) -> Optional[float]:
        """从优先经验回放采样并更新网络(Double DQN + 加权 smooth_l1)"""
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

        if self.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                    next_q = self.target_net(next_states).gather(1, next_actions)
                    target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q
                td_errors = torch.abs(current_q - target_q).detach().float().cpu().numpy()
                elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
                loss = (elementwise_loss * weights.unsqueeze(1)).mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * next_q

            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (elementwise_loss * weights.unsqueeze(1)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.squeeze() + self.eps)
        self._soft_update_target_network()
        return loss.item()

    def _soft_update_target_network(self) -> None:
        """θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)


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


# ====================== 时间窗口缓存 ======================
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _load_or_compute_windows(targets, observer, time_range, cfg) -> dict:
    """计算每个目标的可观测窗口, 返回 {name: [(start_sec, end_sec)]}

    缓存按调度起止时间匹配(±60s);
    """
    start_unix, end_unix = float(time_range[0].unix), float(time_range[1].unix)

    logging.info("Computing available times from scratch")
    avail_times, _ = cal_avail_times(targets, observer, time_range, cfg)
    windows = {
        name: [(float(s.unix) - start_unix, float(e.unix) - start_unix) for s, e in wins]
        for name, wins in avail_times.items()
    }
    return windows


# ====================== 调度结果工具 ======================
def _schedule_key(entries) -> Tuple[int, float, float]:
    """调度质量(字典序): 编排目标数 → 总观测时长 → 总转向时间(取负, 越少越好)"""
    if not entries:
        return (0, 0.0, 0.0)
    return (len({e['target_idx'] for e in entries}),
            sum(e['duration'] for e in entries),
            -sum(e.get('slew_time', 0.0) for e in entries))


def _validate_entries(env: TelescopeEnv, entries) -> list:
    """独立校验调度条目: 窗口包含、总时长、无重叠、转向时间可行, 返回错误列表"""
    errors = []
    ordered = sorted(entries, key=lambda e: e['start_time'])
    prev_end_sec, prev_alt, prev_az = 0.0, env.init_alt, env.init_az
    for e in ordered:
        idx = e['target_idx']
        s, ed = e['start_time'] * 60.0, e['end_time'] * 60.0
        target = env.targets[idx]
        if not any(s >= ws - 1e-6 and ed <= we + 1e-6 for ws, we in target['time_windows']):
            errors.append(f"target {idx} obs [{s / 60:.1f}, {ed / 60:.1f}]min not inside any window")
        if ed > env.obs_duration + 1e-6:
            errors.append(f"target {idx} exceeds total duration")
        alt, az = env._target_altaz(idx, s)
        need = env._slew_between_sec(prev_alt, prev_az, alt, az)
        if s - prev_end_sec < need - 1.0:
            errors.append(f"target {idx} insufficient slew time: "
                          f"gap {s - prev_end_sec:.0f}s < needed {need:.0f}s")
        prev_end_sec, prev_alt, prev_az = ed, *env._target_altaz(idx, ed)
    return errors


def _rollout(agent: Optional[DRLAgent], env: TelescopeEnv, epsilon: float) -> list:
    """用当前策略完整跑一个回合, 返回调度条目列表"""
    state = env.reset()
    done = False
    max_steps = 4 * env.n_targets + 200  # 安全上限, 防止意外死循环
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


def _print_schedule_stats(entries, n_targets: int, tag: str) -> None:
    count, total_obs, neg_slew = _schedule_key(entries)
    logging.info(f"[{tag}] scheduled {count}/{n_targets}, total obs {total_obs:.1f} min, "
                 f"slew {-neg_slew:.1f} min")


# ====================== 训练与调度入口 ======================
def do_schedule(targets, cfg, start_time, end_time, start_az, start_el,
                is_interrupt=False, model_path: Optional[str] = None,
                save_model: bool = False, test_mode: bool = False,
                priorities=None, elevation_quality_weight: float = 1.0, schedule_results_file: str = "schedule_results.json"):
    """使用 DQN 算法进行望远镜观测调度

    Args:
        targets: 目标列表(mock 格式: seq/src/ra/dec/epoch/obs_time)
        cfg: 配置信息
        start_time/end_time: 调度时间窗(UTC 字符串)
        start_az/start_el: 望远镜初始位置(方位角/高度角, 度)
        is_interrupt: 中断标志(也可传入返回 bool 的可调用对象, 每个 episode 后检查)
        model_path: 预训练模型路径(维度匹配时加载)
        save_model: 是否保存训练后的模型
        test_mode: 快速验证模式(减少训练轮次)
        priorities: 每目标科学优先级列表(缺省全 1.0)
        elevation_quality_weight: 仰角质量奖励权重 λ_el

    Returns:
        recs: 调度结果列表, 按 start_time 排序, 每项
              {target_idx, target, slew_time, start_time, end_time, duration, elevation}
              (时间单位为相对调度起点的分钟)
    """
    initialize(cfg)
    site, observer = get_obs_site(cfg)
    named_targets = load_targets(targets, site)
    n_targets = len(named_targets)
    logging.info(f"loaded {n_targets} targets")

    time_range = Time([str(start_time), str(end_time)], scale='utc')
    t0_unix = float(time_range[0].unix)
    total_sec = float(time_range[1].unix - time_range[0].unix)

    if minutes(time_range[0], time_range[1]) < sum([t.obs_time for t in named_targets]):
        raise Exception("Warning: total observation time exceeds scheduling window")

    windows = _load_or_compute_windows(named_targets, observer, time_range, cfg)
    env_targets = [
        {
            'ra': t.sky_coord.ra.degree,
            'dec': t.sky_coord.dec.degree,
            'time_windows': windows[t.name],
            'duration': t.obs_time * 60.0,
        }
        for t in named_targets
    ]

    env = TelescopeEnv(
        site=site,
        targets=env_targets,
        init_altaz=(float(start_el), float(start_az)),
        start_time=t0_unix,
        obs_duration=total_sec,
        priorities=priorities,
        elevation_quality_weight=elevation_quality_weight,
    )

    # 初始化或加载智能体(维度必须匹配)
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
                logging.warning(f"Model dims mismatch "
                                f"(model {loaded.state_dim}/{loaded.action_dim} vs env {env.state_dim}/{env.action_dim}), "
                                f"training from scratch")
        except Exception as e:
            logging.warning(f"Failed to load model: {e}, training from scratch")
    if agent is None:
        agent = DRLAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    logging.info(f"Using device: {agent.device}, mixed precision: {agent.use_mixed_precision}")

    # 训练轮次: 测试模式快速验证, 正式模式充分训练
    attempts = 2 if test_mode else 3
    episodes_per_attempt = 30 if test_mode else 150
    epsilon_start, epsilon_end = 0.9, 0.05
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / episodes_per_attempt)

    best_entries, best_key = [], (-1, -1.0, 0.0)
    model_save_dir = os.path.join(_project_root(), 'models')
    train_started = time.time()
    metrics = []  # 训练曲线数据: 每 episode 一条

    for attempt in range(attempts):
        epsilon = epsilon_start if attempt == 0 else max(epsilon_start * 0.5, epsilon_end)
        logging.info(f"=== Attempt {attempt + 1}/{attempts} start (epsilon={epsilon:.3f}) ===")

        for ep in range(episodes_per_attempt):
            if callable(is_interrupt) and is_interrupt():
                logging.info("interrupted by caller")
                break

            state = env.reset()
            total_reward, done, steps = 0.0, False, 0
            episode_losses = []
            max_steps = 4 * n_targets + 200  # 安全上限, 防止意外死循环

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

            # 记录本回合产生的最优调度
            entries = [dict(e) for e in env.schedule]
            key = _schedule_key(entries)
            if key > best_key:
                best_entries, best_key = entries, key
                _print_schedule_stats(best_entries, n_targets,
                                      f"attempt {attempt + 1} ep {ep} new best (reward {total_reward:.1f})")
                if save_model:
                    save_agent(agent, os.path.join(model_save_dir, 'best_model.pt'))

            # 训练曲线指标
            quality = env.schedule_quality(entries)
            metrics.append({
                'attempt': attempt + 1,
                'episode': attempt * episodes_per_attempt + ep,
                'reward': round(total_reward, 2),
                'coverage': quality['coverage'],
                'weighted_coverage': round(quality['weighted_coverage'], 4),
                'total_obs_min': round(quality['total_obs_min'], 1),
                'total_slew_min': round(quality['total_slew_min'], 2),
                'mean_elevation': round(quality['mean_elevation'], 2),
                'loss': round(float(np.mean(episode_losses)), 5) if episode_losses else None,
                'epsilon': round(epsilon, 4),
            })

            if ep % 20 == 0 or ep == episodes_per_attempt - 1:
                avg_loss = float(np.mean(episode_losses)) if episode_losses else float('nan')
                logging.info(f"Episode {attempt * episodes_per_attempt + ep} | "
                             f"Reward: {total_reward:.2f} | Loss: {avg_loss:.4f} | "
                             f"Epsilon: {epsilon:.3f} | Scheduled: {len(env.schedule)}/{n_targets}")

        if callable(is_interrupt) and is_interrupt():
            break

    logging.info(f"Training finished in {time.time() - train_started:.1f}s, "
                 f"best during training: {best_key[0]}/{n_targets} targets")

    if save_model:
        save_agent(agent, os.path.join(model_save_dir, 'final_model.pt'))

    # ====================== 调度提取 ======================
    # 候选: 贪心 rollout + 少量随机 rollout + 训练期最优, 全部做贪心填补后取最优
    candidates = []
    if best_entries:
        candidates.append(best_entries)

    extract_started = time.time()
    rollout_epsilons = [0.0] + [0.05] * (2 if test_mode else 10)
    for i, eps in enumerate(rollout_epsilons):
        entries = _rollout(agent, env, eps)
        candidates.append(entries)
    logging.info(f"Extraction rollouts done in {time.time() - extract_started:.1f}s")

    filled_candidates = [env.fill_remaining(entries) for entries in candidates]
    # 纯构造式基线(全按稀缺度排序从头编排), 作为覆盖率下限
    filled_candidates.append(env.fill_remaining([]))

    seed_entries, seed_key = [], (-1, -1.0, 0.0)
    for entries in filled_candidates:
        key = _schedule_key(entries)
        if key > seed_key:
            seed_entries, seed_key = entries, key
    _print_schedule_stats(seed_entries, n_targets, "greedy seed")

    # LNS 打磨: 破坏-重建迭代, 提升覆盖率并压缩转向时间
    lns_budget = 15.0 if test_mode else 60.0
    lns_started = time.time()
    final_entries = env.local_search(seed_entries, iterations=100000,
                                     rng=random.Random(), time_budget=lns_budget,
                                     log_every=0)
    logging.info(f"LNS polish done in {time.time() - lns_started:.1f}s "
                 f"({len(filled_candidates)} seeds, best seed {seed_key[0]} targets)")
    final_key = _schedule_key(final_entries)
    _print_schedule_stats(final_entries, n_targets, "final")

    errors = _validate_entries(env, final_entries)
    if errors:
        logging.warning(f"Schedule validation issues ({len(errors)}): {errors[:5]}")
    else:
        logging.info("Schedule validation passed: windows/slew/overlap all feasible")

    if save_model and final_key > best_key:
        best_entries, best_key = final_entries, final_key
        save_agent(agent, os.path.join(model_save_dir, 'best_model.pt'))

    # 组织输出(去掉内部字段, 补充目标名)
    recs = []
    for e in sorted(final_entries, key=lambda x: x['start_time']):
        recs.append({
            'target_idx': e['target_idx'],
            'target': named_targets[e['target_idx']].name,
            'slew_time': round(e['slew_time'], 2),
            'start_time': round(e['start_time'], 2),
            'end_time': round(e['end_time'], 2),
            'duration': round(e['duration'], 2),
            'elevation': round(e.get('el_mean', 0.0), 2),
        })

    start_datetime = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    for rec in recs:
        rec['start_time_str'] = (start_datetime + timedelta(minutes=rec['start_time'])).strftime("%Y-%m-%d %H:%M:%S")
        rec['end_time_str'] = (start_datetime + timedelta(minutes=rec['end_time'])).strftime("%Y-%m-%d %H:%M:%S")
    # 持久化调度结果, 便于下游(服务/可视化)使用
    try:
        import json
        quality = env.schedule_quality(final_entries)
        with open(schedule_results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'start_time': str(start_time),
                'end_time': str(end_time),
                'total_targets': n_targets,
                'scheduled': len(recs),
                'weighted_coverage': round(quality['weighted_coverage'], 4),
                'total_obs_minutes': round(quality['total_obs_min'], 1),
                'total_slew_minutes': round(quality['total_slew_min'], 2),
                'mean_elevation_deg': round(quality['mean_elevation'], 2),
                'records': recs,
            }, f, ensure_ascii=False, indent=2)
        logging.info(f"Schedule results saved to {schedule_results_file}")
    except Exception as e:
        logging.warning(f"Failed to save schedule results: {e}")

    return recs

