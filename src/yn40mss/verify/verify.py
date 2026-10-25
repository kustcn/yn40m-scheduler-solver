"""编排结果独立验证模块。

对 schedule_results.json 做独立复核(不复用调度器内部状态):

1. coverage     每个目标都被调度(源表全集中每个 src 至少出现一次;
                fill-gap 重复观测只要求 >= 1 次)
2. windows      每条观测 [start, end] 完整落入该目标的一个可见性窗口
                (窗口用 cal_avail_times_for_target 独立重算: airmass + 限位,
                分钟粒度, 与调度器同一物理判据)
3. overlap      任意两条观测时间段无重叠(按开始时间排序后相邻比较)
4. consistency  记录自洽: end-start == duration, 起止在调度总窗内,
                slew_time >= 0
5. slew         相邻观测间隔 >= 几何最小转向时间
                (max(d_az/1.0, d_el/0.6) + 15 s, 方位取最短环绕路径;
                位置用 PointModel 修正后的 (az, el), 与调度器同一运动学)

用法:
  from yn40mss.verify import verify_schedule
  report = verify_schedule("schedule_results.json", "targets.json")

  # 命令行
  yn40m-scheduler-solver verify --schedule-results schedule_results.json \
      --targets tests/targets.json
"""
import json
import logging
import os
from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
from astropy.time import Time

from yn40mss.config import DEFAULT_CONFIG_PATH
from yn40mss.core.utils import (
    PointModel,
    cal_avail_times,
    get_obs_site,
    iers_init,
    load_config,
    load_targets,
)

# 与调度器一致的运动学常数(dqn_env.TelescopeEnv)
AZ_RATE_DEG_S = 1.0
EL_RATE_DEG_S = 0.6
SLEW_SETTLE_S = 15.0

# 各类浮点/粒度容差
WINDOW_TOL_S = 1.0     # 窗口边界容差(窗口为分钟粒度)
# 转向时间容差: 调度器用 60 s 网格线性插值 (az, el), 验证器用 astropy
# 精确值, 两者在源快速移动段可差数秒; 容差须覆盖该量化差,
# 而真正的转向违规(少给几十秒)远超此量级.
SLEW_TOL_S = 5.0
DURATION_TOL_MIN = 0.02


def _min_slew_s(az0, el0, az1, el1):
    """两点间几何最小转向时间(秒): 方位最短环绕路径 + 稳定时间."""
    d_az = abs((az1 - az0 + 180.0) % 360.0 - 180.0)
    d_el = abs(el1 - el0)
    return max(d_az / AZ_RATE_DEG_S, d_el / EL_RATE_DEG_S) + SLEW_SETTLE_S


def _pointing(coord, t, site):
    """目标在时刻 t 的 (az, el)(度), 含 PointModel 修正(与调度/绘图一致)."""
    from astropy.coordinates import AltAz

    altaz = coord.transform_to(AltAz(obstime=t, location=site))
    az, el = PointModel(np.asarray(altaz.az.degree, dtype=float),
                        np.asarray(altaz.alt.degree, dtype=float))
    return float(np.asarray(az) % 360.0), float(np.asarray(el))


def verify_schedule(
    schedule_results_file="schedule_results.json",
    targets_file=None,
    config_file=None,
    check_slew=True,
    use_cached_windows=False,
    max_errors=10,
):
    """验证编排结果, 返回结构化报告(dict).

    Parameters
    ----------
    schedule_results_file : str
        调度结果 JSON(含 start_time/end_time/records).
    targets_file : str | None
        源表 JSON(src/ra/dec/obs_time...). None 时回退 tests/targets.json.
    config_file : str | None
        站点配置 YAML. None 时用包内默认配置.
    check_slew : bool
        是否检查相邻观测的转向可行性(默认开).
    use_cached_windows : bool
        True 时允许复用当前目录 target_available_times.pkl 的时间窗缓存
        (仅当时间范围匹配); 默认 False, 即独立重算窗口(完全独立复核).
    max_errors : int
        每类检查最多输出的明细条数.

    Returns
    -------
    dict: {"ok": bool, "checks": {name: {"passed": bool, "errors": [...],
          "summary": str}}, "errors_total": int}
    """
    cfg = load_config(config_file or DEFAULT_CONFIG_PATH)
    site, observer = get_obs_site(cfg)
    try:
        iers_init(cfg)
    except Exception as exc:  # noqa: BLE001 本地 IERS 缺失时回退默认星历表
        print(f"[verify] 本地 IERS 文件加载失败, 使用默认星历表: {exc}")

    with open(schedule_results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") or []
    win_start = str(data.get("start_time") or cfg.common.start_time)
    win_end = str(data.get("end_time") or cfg.common.end_time)

    here = os.path.dirname(os.path.abspath(__file__))
    default_targets = os.path.abspath(
        os.path.join(here, "..", "..", "..", "tests", "targets.json"))
    with open(targets_file or default_targets, "r", encoding="utf-8") as f:
        targets_raw = json.load(f)
    named_targets = load_targets(targets_raw, site)
    by_name = {t.name: t for t in named_targets}

    t0 = Time(win_start, scale="utc")
    t1 = Time(win_end, scale="utc")
    total_sec = float((t1 - t0).to(u.s).value)

    checks = {}

    def add(name, errors, summary):
        checks[name] = {"passed": not errors, "errors": errors[:max_errors],
                        "n_errors": len(errors), "summary": summary}

    # ---------- 1. coverage ----------
    cov_errors = []
    scheduled_names = {r.get("target") for r in records}
    missing = [t.name for t in named_targets if t.name not in scheduled_names]
    if missing:
        cov_errors.append(f"{len(missing)} target(s) never scheduled: "
                          f"{missing[:max_errors]}")
    add("coverage", cov_errors,
        f"{len(scheduled_names)}/{len(named_targets)} targets scheduled "
        f"({len(records)} records, "
        f"{len(records) - len(scheduled_names)} fill-gap extras)")

    # ---------- 独立重算(或复用缓存)可见性窗口 ----------
    time_range = Time([win_start, win_end], scale="utc")
    windows = None
    if use_cached_windows:
        cache = "target_available_times.pkl"
        try:
            import pickle
            with open(cache, "rb") as f:
                cached = pickle.load(f)
            if (isinstance(cached, dict) and cached.get("version") == 2
                    and abs(cached["start_unix"] - float(t0.unix)) < 60
                    and abs(cached["end_unix"] - float(t1.unix)) < 60):
                windows = cached["windows"]
                print(f"[verify] using cached windows from {cache}")
        except Exception:
            windows = None
    if windows is None:
        print("[verify] recomputing visibility windows from scratch "
              "(airmass + elevation limits, 1-min grid)...")
        avail, _ = cal_avail_times(named_targets, observer, time_range, cfg)
        windows = {
            name: [(float(s.unix) - float(t0.unix),
                    float(e.unix) - float(t0.unix)) for s, e in wins]
            for name, wins in avail.items()
        }

    # ---------- 2. windows ----------
    win_errors = []
    for r in records:
        name = r.get("target")
        tgt = by_name.get(name)
        if tgt is None:
            win_errors.append(f"record target '{name}' not in targets table")
            continue
        s_sec = float(r["start_time"]) * 60.0
        e_sec = float(r["end_time"]) * 60.0
        if not any(s_sec >= ws - WINDOW_TOL_S and e_sec <= we + WINDOW_TOL_S
                   for ws, we in windows.get(name, [])):
            win_errors.append(
                f"{name} [{r['start_time']:.1f}, {r['end_time']:.1f}] min "
                f"not inside any visibility window")
    add("windows", win_errors,
        f"{len(records)} records checked against independently "
        f"recomputed windows" if not use_cached_windows or windows is not None
        else "checked against cached windows")

    # ---------- 3. overlap ----------
    ov_errors = []
    ordered = sorted(records, key=lambda r: float(r["start_time"]))
    for a, b in zip(ordered, ordered[1:]):
        if float(b["start_time"]) < float(a["end_time"]) - 1e-6:
            ov_errors.append(
                f"{a.get('target')}[{a['start_time']:.1f}-{a['end_time']:.1f}] "
                f"overlaps {b.get('target')}[{b['start_time']:.1f}-"
                f"{b['end_time']:.1f}] min")
    add("overlap", ov_errors,
        f"{len(ordered)} records, {len(ordered) - 1} adjacent pairs checked")

    # ---------- 4. consistency ----------
    con_errors = []
    for r in records:
        dur = float(r["end_time"]) - float(r["start_time"])
        if abs(dur - float(r["duration"])) > DURATION_TOL_MIN:
            con_errors.append(f"{r.get('target')}: end-start {dur:.2f} min != "
                              f"duration {float(r['duration']):.2f} min")
        if float(r["start_time"]) < -1e-6 or float(r["end_time"]) * 60.0 > total_sec + 1e-6:
            con_errors.append(f"{r.get('target')}: [{r['start_time']:.1f}, "
                              f"{r['end_time']:.1f}] min outside scheduling window")
        if float(r.get("slew_time", 0.0)) < -1e-9:
            con_errors.append(f"{r.get('target')}: negative slew_time")
    add("consistency", con_errors,
        f"duration/window-bounds/slew-sign checked on {len(records)} records")

    # ---------- 5. slew feasibility ----------
    slew_errors = []
    if check_slew:
        prev_az, prev_el = float(cfg.common.start_az), float(cfg.common.start_el)
        prev_end_sec = 0.0
        for r in ordered:
            name = r.get("target")
            tgt = by_name.get(name)
            if tgt is None:
                continue
            s_sec = float(r["start_time"]) * 60.0
            az1, el1 = _pointing(tgt.sky_coord, t0 + s_sec * u.s, site)
            gap = s_sec - prev_end_sec
            need = _min_slew_s(prev_az, prev_el, az1, el1)
            if gap + SLEW_TOL_S < need:
                slew_errors.append(
                    f"before {name}[{r['start_time']:.1f} min]: gap "
                    f"{gap:.0f} s < required slew {need:.0f} s")
            e_sec = float(r["end_time"]) * 60.0
            prev_az, prev_el = _pointing(tgt.sky_coord, t0 + e_sec * u.s, site)
            prev_end_sec = e_sec
        add("slew", slew_errors,
            f"adjacent-pointing slew feasibility "
            f"(rate {AZ_RATE_DEG_S}/{EL_RATE_DEG_S} deg/s + "
            f"{SLEW_SETTLE_S:.0f} s settle)")
    else:
        add("slew", [], "skipped (--no-slew-check)")

    errors_total = sum(c["n_errors"] for c in checks.values())
    ok = all(c["passed"] for c in checks.values())

    # ---------- 人类可读报告 ----------
    print("\n" + "=" * 64)
    print("SCHEDULE VERIFICATION REPORT")
    print("=" * 64)
    print(f"results : {schedule_results_file}")
    print(f"targets : {targets_file or default_targets}")
    print(f"window  : {win_start} -> {win_end} "
          f"({total_sec / 60.0:.0f} min)")
    for name, c in checks.items():
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"[{mark}] {name:<13} {c['summary']}")
        for e in c["errors"]:
            print(f"        - {e}")
    print("-" * 64)
    verdict = "ALL CHECKS PASSED" if ok else f"{errors_total} violation(s) found"
    print(f"{'✅' if ok else '❌'} {verdict}")
    print("=" * 64)

    logging.info("verify_schedule: %s (%d violations)", verdict, errors_total)
    return {"ok": ok, "checks": checks, "errors_total": errors_total}
