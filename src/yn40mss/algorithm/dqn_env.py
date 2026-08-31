import random
import logging

import numpy as np
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u


class TelescopeEnv:
    """望远镜观测调度强化学习环境(为 DQN 设计)

    设计要点:
    - 内部时间离散为相对调度起点的秒;alt/az 按 grid_step(60s)网格在构造时
      一次性向量化预计算,episode 内 O(1) 查表,避免每步调用 astropy 变换
    - 每个目标在状态向量中占据固定位置,网络输出第 i 维 Q 值恒定对应目标 i
    - 可观测性完全由预计算时间窗口决定(窗口已含 airmass 与仰角限位约束),
      要求 [t+slew, t+slew+duration] 完整落入同一窗口
    - 转向时间采用真实转速的运动学模型(方位 1.0°/s、俯仰 0.6°/s,与
      utils.azel_slew_time_to_this 一致)+ 稳定时间,方位角按最短路径环绕
    - 成功观测时把调度条目记录到 self.schedule(start_time/end_time/duration
      单位为分钟,可直接用于输出)
    """

    OBS_FEAT_DIM = 4      # 全局特征: [时间进度, az/360, alt/90, 剩余目标比例]
    TARGET_FEAT_DIM = 6   # 每目标特征: [alt/90, az/360, duration归一, completed, 可开始, 紧迫度]

    def __init__(self,
                 site: EarthLocation,
                 targets,
                 init_altaz,
                 start_time,
                 obs_duration,
                 max_az_speed=1.0,    # 方位转速 (度/秒), 真实值
                 max_alt_speed=0.6,   # 俯仰转速 (度/秒), 真实值
                 slew_settle=15.0,    # 每次转向的稳定时间 (秒)
                 grid_step=60.0,      # alt/az 预计算网格步长 (秒)
                 time_step=60.0,      # 时间推进最小步长 (秒)
                 priorities=None,             # 每目标科学优先级 (缺省全 1.0)
                 elevation_quality_weight=1.0):  # 仰角质量奖励权重 λ_el
        """
        targets: [{
            'ra': 赤经 (度),
            'dec': 赤纬 (度),
            'time_windows': [(start_sec, end_sec), ...],  # 相对调度起点的秒
            'duration': 观测时长 (秒)
        }]
        init_altaz: 初始位置 (高度角, 方位角) 单位: 度
        start_time: 调度起点 unix 时间戳 (秒)
        obs_duration: 总调度时长 (秒)
        """
        self.site = site
        self.targets = targets
        self.start_time = float(start_time)
        self.init_alt = float(init_altaz[0])
        self.init_az = float(init_altaz[1]) % 360.0
        self.obs_duration = float(obs_duration)
        self.max_az_speed = max_az_speed
        self.max_alt_speed = max_alt_speed
        self.slew_settle = slew_settle
        self.grid_step = grid_step
        self.time_step = time_step
        self.priorities = list(priorities) if priorities is not None else [1.0] * len(targets)
        self.elevation_quality_weight = float(elevation_quality_weight)

        self.n_targets = len(targets)
        self.state_dim = self.OBS_FEAT_DIM + self.TARGET_FEAT_DIM * self.n_targets
        self.action_dim = self.n_targets
        self.observation_space = self.state_dim  # 兼容旧接口
        self.min_duration = min(t['duration'] for t in targets) if targets else 0.0

        self._precompute_altaz_grid()

        # 运行时状态
        self.current_time = 0.0
        self.completed = set()
        self.schedule = []
        self.altaz = (self.init_alt, self.init_az)
        self.last_target_idx = None

    # ====================== 预计算 ======================
    def _precompute_altaz_grid(self):
        """把所有目标在调度时长内的 alt/az 按 grid_step 网格一次性算好"""
        n_grid = int(self.obs_duration // self.grid_step) + 2
        self.n_grid = n_grid
        times = Time(self.start_time, format='unix', scale='utc') \
            + np.arange(n_grid) * self.grid_step * u.second
        ras = np.array([t['ra'] for t in self.targets]) * u.deg
        decs = np.array([t['dec'] for t in self.targets]) * u.deg
        coords = SkyCoord(ra=ras, dec=decs)
        aa = coords.transform_to(AltAz(obstime=times[:, None], location=self.site))
        self.grid_alt = aa.alt.degree   # [n_grid, N]
        self.grid_az = aa.az.degree % 360.0

    def _grid_index(self, t_sec):
        pos = t_sec / self.grid_step
        i0 = int(np.clip(np.floor(pos), 0, self.n_grid - 2))
        frac = float(min(max(pos - i0, 0.0), 1.0))
        return i0, frac

    def _target_altaz(self, idx, t_sec):
        """目标 idx 在 t_sec 时刻的 (alt, az), 网格线性插值(处理方位环绕)"""
        i0, frac = self._grid_index(t_sec)
        alt = self.grid_alt[i0, idx] * (1 - frac) + self.grid_alt[i0 + 1, idx] * frac
        az0 = self.grid_az[i0, idx]
        az1 = self.grid_az[i0 + 1, idx]
        az = az0 + ((az1 - az0 + 180.0) % 360.0 - 180.0) * frac
        return float(alt), float(az % 360.0)

    # ====================== 转向模型 ======================
    def _slew_time_sec(self, from_alt, from_az, idx, t_arrive):
        """从 (from_alt, from_az) 转到目标 idx 在 t_arrive 时刻位置所需时间(秒)

        方位角取最短环绕路径
        """
        to_alt, to_az = self._target_altaz(idx, t_arrive)
        d_az = abs((to_az - from_az + 180.0) % 360.0 - 180.0)
        d_el = abs(to_alt - from_alt)
        return max(d_az / self.max_az_speed, d_el / self.max_alt_speed) + self.slew_settle

    def _slew_time_to_idx(self, idx, t_start):
        """从当前望远镜位置转向目标 idx 的转向时间(秒), 目标位置随到达时刻变化, 迭代自洽"""
        alt, az = self.altaz
        slew = self._slew_time_sec(alt, az, idx, t_start)
        return self._slew_time_sec(alt, az, idx, t_start + slew)

    def _slew_times_all(self, t_sec):
        """当前时刻转向所有目标的转向时间估计(秒), 用于状态特征与掩码"""
        return np.array([self._slew_time_to_idx(i, t_sec) for i in range(self.n_targets)])

    def _slew_between_sec(self, from_alt, from_az, to_alt, to_az):
        """两个指向之间的转向时间(秒)"""
        d_az = abs((to_az - from_az + 180.0) % 360.0 - 180.0)
        d_el = abs(to_alt - from_alt)
        return max(d_az / self.max_az_speed, d_el / self.max_alt_speed) + self.slew_settle

    # ====================== 可行性 ======================
    def _containing_window(self, idx, obs_start, obs_end):
        """返回完全包含 [obs_start, obs_end] 的窗口 (start, end), 无则 None"""
        for s, e in self.targets[idx]['time_windows']:
            if obs_start >= s - 1e-6 and obs_end <= e + 1e-6:
                return (s, e)
        return None

    def _startable(self, idx, t_sec, slew_sec=None):
        """目标 idx 在 t_sec 时刻能否立即开始(转向+观测完整落入窗口与总时长)"""
        if idx in self.completed:
            return False
        target = self.targets[idx]
        if slew_sec is None:
            slew_sec = self._slew_time_to_idx(idx, t_sec)
        arrive = t_sec + slew_sec
        end = arrive + target['duration']
        if end > self.obs_duration:
            return False
        return self._containing_window(idx, arrive, end) is not None

    def _future_feasible(self, idx, t_sec, slew_sec=None):
        """目标 idx 在 t_sec 之后是否还存在可完成观测的窗口"""
        if idx in self.completed:
            return False
        target = self.targets[idx]
        if slew_sec is None:
            slew_sec = self._slew_time_to_idx(idx, t_sec)
        for s, e in target['time_windows']:
            if e >= t_sec + slew_sec + target['duration'] - 1e-6:
                return True
        return False

    def _window_slack(self, idx, t_sec, slew_sec):
        """最近可行窗口的松弛时间(秒): 窗口结束 - 预计观测结束; 不可行为 None"""
        target = self.targets[idx]
        arrive = t_sec + slew_sec
        end = arrive + target['duration']
        best = None
        for s, e in target['time_windows']:
            if end <= e + 1e-6 and end >= s:
                slack = e - end
                if best is None or slack < best:
                    best = slack
        return best

    def _no_more_feasible(self):
        """是否所有未完成目标都已不可能再完成观测"""
        slews = None
        for i in range(self.n_targets):
            if i in self.completed:
                continue
            if slews is None:
                slews = self._slew_times_all(self.current_time)
            if self._future_feasible(i, self.current_time, slews[i]):
                return False
        return True

    def _get_valid_actions(self):
        """当前可立即执行的合法动作列表(未完成且现在就能开始观测)"""
        slews = self._slew_times_all(self.current_time)
        return [i for i in range(self.n_targets)
                if self._startable(i, self.current_time, slews[i])]

    # ====================== 状态 ======================
    def _get_state(self):
        t = self.current_time
        slews = self._slew_times_all(t)
        global_feats = np.array([
            t / self.obs_duration,
            self.altaz[1] / 360.0,
            self.altaz[0] / 90.0,
            (self.n_targets - len(self.completed)) / self.n_targets,
        ], dtype=np.float32)

        per_target = np.zeros((self.n_targets, self.TARGET_FEAT_DIM), dtype=np.float32)
        for i, target in enumerate(self.targets):
            alt, az = self._target_altaz(i, t)
            done_flag = 1.0 if i in self.completed else 0.0
            startable = 1.0 if (done_flag == 0.0 and self._startable(i, t, slews[i])) else 0.0
            if done_flag:
                slack_norm = 1.0
            else:
                slack = self._window_slack(i, t, slews[i])
                slack_norm = 1.0 if slack is None else float(min(slack / self.obs_duration, 1.0))
            per_target[i] = (alt / 90.0, az / 360.0,
                             target['duration'] / 3600.0,
                             done_flag, startable, slack_norm)
        return np.concatenate([global_feats, per_target.flatten()])

    # ====================== 回合控制 ======================
    def reset(self):
        self.current_time = 0.0
        self.completed = set()
        self.schedule = []
        self.altaz = (self.init_alt, self.init_az)
        self.last_target_idx = None
        return self._get_state()

    def _next_event_time(self):
        """下一次有目标可以开始观测的时刻(秒); 没有则 None"""
        slews = self._slew_times_all(self.current_time)
        best = None
        for i in range(self.n_targets):
            if i in self.completed:
                continue
            slew = slews[i]
            dur = self.targets[i]['duration']
            for s, e in self.targets[i]['time_windows']:
                if e < self.current_time + slew + dur - 1e-6:
                    continue  # 该窗口剩余时间不够完成观测
                candidate = max(self.current_time + self.time_step, s - slew)
                if best is None or candidate < best:
                    best = candidate
        return best

    def _elevation_quality(self, idx, obs_start, obs_end):
        """观测期间的平均高度角(度), 取起点/中点/终点网格均值"""
        alt_a, _ = self._target_altaz(idx, obs_start)
        alt_m, _ = self._target_altaz(idx, 0.5 * (obs_start + obs_end))
        alt_b, _ = self._target_altaz(idx, obs_end)
        return (alt_a + alt_m + alt_b) / 3.0

    def _terminal_reward(self, reward):
        total_p = float(sum(self.priorities))
        done_p = float(sum(self.priorities[i] for i in self.completed))
        reward += 50.0 * (done_p / total_p if total_p > 0 else len(self.completed) / self.n_targets)
        if len(self.completed) == self.n_targets:
            reward += 100.0
        return reward

    def step(self, action):
        reward = 0.0
        info = {}

        # ---- 等待/推进动作: 跳到下一个有目标可开始的时刻 ----
        if action is None or action < 0 or action >= self.n_targets:
            next_t = self._next_event_time()
            if next_t is None:
                # 再无任何目标可以观测, 回合结束
                reward = self._terminal_reward(reward)
                self.last_target_idx = -1
                return self._get_state(), reward, True, {'ended': 'no_more_feasible'}
            advanced = next_t - self.current_time
            self.current_time = next_t
            reward -= 0.02 * advanced / 60.0  # 时间流逝的小惩罚
            self.last_target_idx = -1
            return self._get_state(), reward, False, {'advanced_min': advanced / 60.0}

        # ---- 观测动作 ----
        try:
            target = self.targets[action]
            slew = self._slew_time_to_idx(action, self.current_time)
            arrive = self.current_time + slew
            end = arrive + target['duration']

            window = self._containing_window(action, arrive, end)
            if window is None or end > self.obs_duration:
                # 动作掩码正常时不会走到这里; 兜底: 不消费目标, 时间小幅推进
                reward = -2.0
                self.current_time = min(self.current_time + self.time_step, self.obs_duration)
                done = self.current_time >= self.obs_duration or self._no_more_feasible()
                if done:
                    reward = self._terminal_reward(reward)
                info['invalid_action'] = action
                self.last_target_idx = -1
                return self._get_state(), reward, done, info

            # 记录调度条目(时间为分钟, 与输出格式一致)
            alt_at_start, az_at_start = self._target_altaz(action, arrive)
            exit_alt, exit_az = self._target_altaz(action, end)
            el_mean = self._elevation_quality(action, arrive, end)
            entry = {
                'target_idx': action,
                'slew_time': slew / 60.0,
                'start_time': arrive / 60.0,
                'end_time': end / 60.0,
                'duration': target['duration'] / 60.0,
                'el_mean': el_mean,
                'alt': alt_at_start,       # 观测开始时目标位置(转向终点)
                'az': az_at_start,
                'enter_alt': self.altaz[0],  # 转向前望远镜位置
                'enter_az': self.altaz[1],
                'exit_alt': exit_alt,        # 观测结束时目标位置(望远镜跟随)
                'exit_az': exit_az,
            }
            self.schedule.append(entry)
            self.completed.add(action)
            self.altaz = (exit_alt, exit_az)
            self.current_time = end
            self.last_target_idx = action

            # ---- 多目标奖励塑形 ----
            dur_min = target['duration'] / 60.0
            priority = float(self.priorities[action])
            # 目标1: 科学产出(按目标优先级加权的观测时长)
            reward = 2.0 * priority * dur_min
            # 目标2: 紧凑度(转向时间惩罚)
            reward -= 0.5 * (slew / 60.0)
            # 目标3: 观测质量(高仰角观测的加成, 归一化到最大限位仰角)
            reward += self.elevation_quality_weight * (el_mean / 65.0) * dur_min
            # 目标4: 紧迫性(窗口即将关闭的优先)
            slack = (window[1] - end) / 60.0
            if slack < 120.0:
                reward += 2.0

            done = len(self.completed) == self.n_targets \
                or self.current_time >= self.obs_duration - self.min_duration \
                or self._no_more_feasible()
            if done:
                reward = self._terminal_reward(reward)
            info['slew_time'] = slew / 60.0
            return self._get_state(), reward, done, info

        except Exception as e:  # 兜底: 环境异常不应让训练崩溃
            return self._get_state(), -5.0, False, {'error': str(e)}

    # ====================== 贪心填补 ======================
    def fill_remaining(self, schedule=None, rng=None, order_noise=0.0):
        """把未编排目标按窗口稀缺度(最约束优先)贪心插入空闲时隙

        - 稀缺度 = 最短可用窗口(可容纳完整观测)的剩余时长, 越小越优先;
          窗口碎片(短于观测时长)不参与度量
        - order_noise > 0 时对排序叠加均匀随机扰动(配合 rng 可做多重启搜索)
        - 插入不移动已有条目: 需满足 从前一位置转向的时间、以及给下一条目
          留出从本目标转过去的转向时间; 因此结果永不重叠、永不使既有条目失效
        - schedule 为 None/空 时即"按稀缺度顺序的纯构造式调度"
        返回新的调度条目列表(分钟), 已按 start_time 排序
        """
        entries = [dict(e) for e in (schedule or [])]
        entries.sort(key=lambda e: e['start_time'])
        placed = {e['target_idx'] for e in entries}

        def scarcity(i):
            dur = self.targets[i]['duration']
            spans = [e - s for s, e in self.targets[i]['time_windows'] if e - s >= dur - 1e-6]
            base = (min(spans) - dur) if spans else float('inf')
            if order_noise > 0 and rng is not None:
                base += rng.uniform(0.0, order_noise)
            return base

        remaining = sorted((i for i in range(self.n_targets) if i not in placed), key=scarcity)
        for i in remaining:
            self._try_insert(entries, i)
        return entries

    def local_search(self, entries, iterations=1000, rng=None, order_noise=1800.0,
                     time_budget=30.0, log_every=200):
        """大邻域搜索(LNS): 反复"破坏-重建"提升调度覆盖率与紧凑度

        - 破坏: 随机移除 k 个已编排条目
        - 重建: 用带扰动稀缺度排序把所有未编排目标(含被移除的)重新贪心插入
        - 接受准则: 编排数 → 总观测时长 → 总转向时间(更少) 的字典序
        """
        import time as _time
        if rng is None:
            rng = random.Random()
        if not entries:
            entries = self.fill_remaining([], rng=rng, order_noise=order_noise)

        def key(es):
            if not es:
                return (-1, -1.0, 0.0)
            return (len({e['target_idx'] for e in es}),
                    sum(e['duration'] for e in es),
                    -sum(e.get('slew_time', 0.0) for e in es))

        best = [dict(e) for e in entries]
        best_key = key(best)
        cur, cur_key = best, best_key
        started = _time.time()
        for it in range(iterations):
            if _time.time() - started > time_budget:
                break
            placed_idx = [e['target_idx'] for e in cur]
            if not placed_idx:
                break
            k = rng.randint(2, min(8, len(placed_idx)))
            victims = set(rng.sample(placed_idx, k))
            partial = [e for e in cur if e['target_idx'] not in victims]
            cand = self.fill_remaining(partial, rng=rng, order_noise=order_noise)
            cand_key = key(cand)
            if cand_key > cur_key:
                cur, cur_key = cand, cand_key
                if cand_key > best_key:
                    best, best_key = [dict(e) for e in cand], cand_key
                    logging.info(f"[LNS it {it}] improved: {best_key[0]} targets, "
                                 f"{best_key[1]:.1f} min obs, slew {-best_key[2]:.1f} min")
            if log_every and (it + 1) % log_every == 0:
                logging.info(f"[LNS it {it}] current {cur_key[0]} targets, best {best_key[0]}")
        return best

    def _try_insert(self, entries, idx):
        """尝试把目标 idx 插入 entries 的某个空隙, 成功就地插入并返回 True"""
        target = self.targets[idx]
        dur_min = target['duration'] / 60.0
        total_min = self.obs_duration / 60.0

        # 空隙: (gap_start, next_start, next_entry_or_None, prev_pos)
        gaps = []
        free_start = 0.0
        prev_pos = (self.init_alt, self.init_az)
        for e in entries:
            gaps.append((free_start, e['start_time'], e, prev_pos))
            free_start = e['end_time']
            prev_pos = (e['exit_alt'], e['exit_az'])
        gaps.append((free_start, total_min, None, prev_pos))

        for gap_start, next_start, next_entry, prev_pos in gaps:
            for ws, we in target['time_windows']:
                ws_min, we_min = ws / 60.0, we / 60.0
                # 最早观测开始: 空隙开始 + 从前一位置转向的时间(迭代两次自洽)
                base = max(gap_start, ws_min)
                for _ in range(2):
                    slew_prev_min = self._slew_time_sec(prev_pos[0], prev_pos[1], idx, base * 60.0) / 60.0
                    base = max(gap_start + slew_prev_min, ws_min)
                obs_start = base
                if obs_start < gap_start - 1e-9 or obs_start < ws_min - 1e-9:
                    continue
                obs_end = obs_start + dur_min

                # 观测结束的上限: 窗口结束 / 总时长 / 给下一条目留出转向时间
                obs_end_limit = min(we_min, total_min)
                if next_entry is not None:
                    src_alt, src_az = self._target_altaz(idx, obs_end * 60.0)
                    slew_next_min = self._slew_between_sec(src_alt, src_az,
                                                           next_entry['alt'], next_entry['az']) / 60.0
                    obs_end_limit = min(obs_end_limit, next_start - slew_next_min)

                if obs_end <= obs_end_limit + 1e-9:
                    entry = {
                        'target_idx': idx,
                        'slew_time': slew_prev_min,
                        'start_time': obs_start,
                        'end_time': obs_end,
                        'duration': dur_min,
                        'el_mean': self._elevation_quality(idx, obs_start * 60.0, obs_end * 60.0),
                        'alt': self._target_altaz(idx, obs_start * 60.0)[0],
                        'az': self._target_altaz(idx, obs_start * 60.0)[1],
                        'enter_alt': prev_pos[0],
                        'enter_az': prev_pos[1],
                        'exit_alt': self._target_altaz(idx, obs_end * 60.0)[0],
                        'exit_az': self._target_altaz(idx, obs_end * 60.0)[1],
                    }
                    entries.append(entry)
                    entries.sort(key=lambda e: e['start_time'])
                    return True
        return False

    def schedule_quality(self, entries) -> dict:
        """调度质量指标(供训练日志与实验评估统一使用)

        coverage: 编排目标数; weighted_coverage: 优先级加权覆盖率;
        mean_elevation: 时长加权平均观测仰角
        """
        if not entries:
            return {'coverage': 0, 'weighted_coverage': 0.0, 'total_obs_min': 0.0,
                    'total_slew_min': 0.0, 'mean_elevation': 0.0}
        total_p = float(sum(self.priorities))
        done_p = float(sum(self.priorities[e['target_idx']] for e in entries))
        total_obs = float(sum(e['duration'] for e in entries))
        total_slew = float(sum(e.get('slew_time', 0.0) for e in entries))
        mean_el = (sum(e['duration'] * e.get('el_mean', 0.0) for e in entries) / total_obs
                   if total_obs > 0 else 0.0)
        return {
            'coverage': len({e['target_idx'] for e in entries}),
            'weighted_coverage': (done_p / total_p) if total_p > 0 else 1.0,
            'total_obs_min': total_obs,
            'total_slew_min': total_slew,
            'mean_elevation': mean_el,
        }
