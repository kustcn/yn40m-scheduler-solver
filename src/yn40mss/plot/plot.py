"""调度结果绘图模块(出版级样式)。

将调度结果 (schedule_results.json) 绘制为 5 幅图, 按 Nature 系出版规范输出
(PDF + 高分辨率 PNG, 统一色彩族, 每个面板服务一个结论):
  1. fig1_slew_time.png                : 每次换源时间(单面板柱状图), 越限段标红, 叠加预估上限参考线
  2. fig2_start_end_el.png             : 每段观测开始/结束仰角(单面板双序列), 叠加限位带
  3. fig3_el_limit_check.png           : a. 段内相对时间 / b. 相对起始的绝对时间, 源仰角轨迹 + 限位线
  4. fig4_unique_source_coverage.png   : 累计观测时长 vs 已覆盖的不同源数(step 曲线, hero 面板)
  5. fig5_az_el_strip.png              : 每段观测起始点在展开 az 轴上的 (az, el) 轨迹, 限位带 + 方向箭头

用法:
  # 作为模块导入
  from yn40mss.plot import plot_schedule_results
  plot_schedule_results("schedule_results.json", save_dir="out", show=False)

  # 命令行直接运行
  python src/yn40mss/plot/plot.py --schedule-results schedule_results.json --save-dir out
"""
import argparse
import json
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

# ---- 保证脚本可独立运行: python src/yn40mss/plot/plot.py ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # .../src
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import matplotlib.pyplot as plt

from yn40mss.config import DEFAULT_CONFIG_PATH
from yn40mss.core.utils import load_config, get_obs_site, PointModel, iers_init

# ---- 出版级绘图参数(先于任何图形创建) ----
# 可编辑文本: SVG 保留 <text> 节点, PDF 嵌入 TrueType
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
# 紧凑期刊版面: 正文 8pt, 去上/右边框, 刻度向内
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# ---- 统一色彩族: 主序列(蓝) / 信号(红, 仅用于越限等方向性提示) / 中性灰(参考) ----
# 蓝/红均为中深色, 灰度打印仍可区分
_C_MAIN = "#0F4D92"      # 主序列
_C_SEC = "#3775BA"       # 次级序列(淡, 用于多线轨迹)
_C_RED = "#B64342"       # 越限 / 终值 / 方向提示
_C_NEU = "#767676"       # 参考线 / 说明文字
_C_NEU_DARK = "#4D4D4D"  # 强参考线


def _panel_label(ax, label, x=-0.09, y=1.02, fontsize=9):
    """多面板图的面板标签(小写加粗, 靠近左上)."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom")


def _light_grid(ax):
    ax.grid(True, ls=":", lw=0.5, color="0.85")


def _constraint_band(ax, el_lim, el_max):
    """叠加限位带: 允许区(浅灰底) + 禁带(淡红底) + 限位线.

    同时把 Y 轴固定为 0~80 deg, 并将上下限角度(如 10° / 65°)作为
    Y 轴刻度显示, 使限位值在 Y 轴上直接可读.
    """
    ax.axhspan(0.0, el_lim, color=_C_RED, alpha=0.08, zorder=0)
    ax.axhspan(el_max, 80.0, color=_C_RED, alpha=0.08, zorder=0)
    ax.axhspan(el_lim, el_max, color="0.93", zorder=0)
    ax.axhline(el_lim, color=_C_NEU_DARK, ls="--", lw=1, zorder=3)
    ax.axhline(el_max, color=_C_NEU_DARK, ls="--", lw=1, zorder=3)
    ax.set_ylim(0.0, 80.0)
    ax.set_yticks(np.unique(np.append(np.arange(0.0, 81.0, 10.0), [el_lim, el_max])))


def _finish(fig):
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.8)


def src_az_el(skycoord, time, site):
    """返回 (az, el) 数组(度): AltAz 变换 + PointModel 指向模型修正。

    Parameters
    ----------
    skycoord : astropy.coordinates.SkyCoord
        源的天球坐标 (ICRS, 赤经为 hourangle 量纲).
    time : astropy.time.Time (标量或数组)
        观测时刻 (UTC).
    site : astropy.coordinates.EarthLocation
        观测站位置.

    Returns
    -------
    (az, el) : (np.ndarray, np.ndarray)
        方位角(回绕到 [0, 360)) 与仰角(度), 仰角不做任何修改.
    """
    altaz = skycoord.transform_to(AltAz(obstime=time, location=site))
    az = np.asarray(altaz.az.degree, dtype=float)
    el = np.asarray(altaz.alt.degree, dtype=float)
    az, el = PointModel(az, el)
    az = np.asarray(az, dtype=float) % 360.0
    return az, np.asarray(el, dtype=float)


def _load_targets_table(targets_file):
    """读取源表 JSON (list of {src, ra, dec, ...}), 返回 {src: {"ra":.., "dec":..}}."""
    if not targets_file or not os.path.isfile(targets_file):
        return {}
    with open(targets_file, "r", encoding="utf-8") as f:
        table = json.load(f)
    result = {}
    for item in table:
        name = item.get("src") or item.get("name")
        if name and "ra" in item and "dec" in item:
            result[name] = item
    return result


def _resolve_coord(record, targets_table):
    """解析单条记录的源坐标: 优先用记录内嵌 ra/dec, 否则从源表按目标名匹配."""
    name = record.get("target", "")
    if "ra" in record and "dec" in record:
        try:
            return SkyCoord(f"{record['ra']} {record['dec']}", unit=(u.hourangle, u.deg))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"record '{name}' 内嵌 ra/dec 无法解析: {record['ra']} {record['dec']}"
            ) from exc
    item = targets_table.get(name)
    if item is None:
        raise ValueError(
            f"record '{name}' 未内嵌 ra/dec, 且在 targets 表中找不到对应源(检查 --targets 文件)"
        )
    return SkyCoord(f"{item['ra']} {item['dec']}", unit=(u.hourangle, u.deg))


def plot_schedule_results(
    schedule_results_file="schedule_results.json",
    targets_file=None,
    config_file=None,
    save_dir=None,
    show=True,
    dpi=300,
):
    """绘制调度结果 5 幅图(出版级样式), 支持保存为 PDF/PNG.

    Parameters
    ----------
    schedule_results_file : str
        调度结果 JSON 路径, 结构含 start_time 与 records
        (每条含 target / slew_time(min) / start_time_str / end_time_str / duration(min)).
    targets_file : str | None
        源表 JSON 路径 (字段 src/ra/dec). None 时: 优先使用 records 内嵌 ra/dec,
        否则回退到项目根 tests/targets.json.
    config_file : str | None
        站点配置 YAML 路径. None 时使用 yn40mss.config.DEFAULT_CONFIG_PATH.
    save_dir : str | None
        非 None 时创建目录并保存 5 张图, 每张输出 .pdf + .png(dpi 可配),
        bbox_inches='tight'.
    show : bool
        是否弹窗显示 (plt.show()).
    dpi : int
        保存 PNG 的分辨率.

    Returns
    -------
    dict
        {"figs": [fig1, ..., fig5], "paths": [主文件路径列表(PNG), 若无保存则为 []]}
    """
    # ---------- 配置 / 站点 / IERS ----------
    cfg_path = config_file or DEFAULT_CONFIG_PATH
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(cfg_path)
    cfg = load_config(cfg_path)
    site, _ = get_obs_site(cfg)
    try:
        iers_init(cfg)
    except Exception as exc:  # noqa: BLE001 本地 IERS 文件缺失时静默回退默认星历表
        print(f"[plot] 本地 IERS 文件加载失败, 使用默认星历表: {exc}")

    # ---------- 读调度结果 ----------
    with open(schedule_results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") or []
    if not records:
        print("[plot] 调度结果中没有观测记录 (records 为空), 无图可绘")
        return {"figs": [], "paths": []}
    print(f"[plot] 观测记录数: {len(records)}")

    # ---------- 源坐标表 ----------
    default_targets = os.path.join(_PROJ_ROOT, "tests", "targets.json")
    targets_table = _load_targets_table(targets_file or default_targets)
    sky_coords = [_resolve_coord(r, targets_table) for r in records]

    # ---------- 数据准备 ----------
    index = np.arange(len(records))
    # 换源时间: JSON 单位为分钟, 绘图单位换算为秒(与预估上限 6.4 min = 384 s 对齐)
    slew_plot = np.array([float(r.get("slew_time", 0.0)) * 60.0 for r in records])

    # 每段观测起止仰角
    el_start_plot = np.empty(len(records))
    el_end_plot = np.empty(len(records))
    for i, (r, coord) in enumerate(zip(records, sky_coords)):
        t_start = Time(r["start_time_str"])
        t_end = Time(r["end_time_str"])
        _, el_start_plot[i] = src_az_el(coord, t_start, site)
        _, el_end_plot[i] = src_az_el(coord, t_end, site)

    el_lim = float(cfg.common.elevating_lim)
    el_max = float(cfg.common.elevating_max)
    slew_limit_s = float(cfg.common.slew_time) * 60.0

    # ---------- 图1: 换源时间(单面板, 越限段标红) ----------
    n_exceed = int((slew_plot > slew_limit_s).sum())
    fig1 = plt.figure(1, figsize=(7.2, 3.0))
    ax0 = fig1.add_subplot(111)
    ax0.bar(index, slew_plot, 0.6, color=_C_MAIN, label="slew time")
    if n_exceed:
        ax0.bar(index[slew_plot > slew_limit_s], slew_plot[slew_plot > slew_limit_s],
                0.6, color=_C_RED, label=f"exceeds {cfg.common.slew_time} min estimate")
    ax0.axhline(slew_limit_s, color=_C_NEU_DARK, ls="--", lw=1,
                label=f"estimate limit {cfg.common.slew_time} min")
    ax0.set_xlabel("Source index")
    ax0.set_ylabel("Slew time (s)")
    ax0.set_xlim(-0.6, len(records) - 0.4)
    ax0.set_ylim(bottom=0.0)
    ax0.legend(loc=[0.7, 0.8], fontsize=9)
    # ax0.text(0.01, 0.02,
    #          f"n = {len(records)} switches; {n_exceed} exceed the {cfg.common.slew_time} min estimate",
    #          transform=ax0.transAxes, ha="left", va="bottom", fontsize=7, color=_C_NEU)
    _light_grid(ax0)
    _finish(fig1)

    # ---------- 图2: 起止仰角(单面板双序列, 叠加限位带) ----------
    fig2 = plt.figure(2, figsize=(7.2, 3.0))
    # fig2.suptitle("Start and end elevation of each observation", y=0.98)
    ax2 = fig2.add_subplot(111)
    _constraint_band(ax2, el_lim, el_max)
    ax2.plot(index, el_start_plot, "o-", ms=3, lw=1, color=_C_MAIN, label="start elevation")
    ax2.plot(index, el_end_plot, "s-", ms=3, lw=1, color=_C_RED, label="end elevation")
    ax2.set_xlabel("Target Index")
    ax2.set_ylabel("Elevation (deg)")
    ax2.set_xlim(-0.6, len(records) - 0.4)
    ax2.legend(loc=[0.7, 0.8], fontsize=9)
    # ax2.text(0.01, 0.98, f"n = {len(records)} targets; all within {el_lim:.0f}\u2013{el_max:.0f} deg",
    #          transform=ax2.transAxes, ha="left", va="top", fontsize=7, color=_C_NEU)
    _light_grid(ax2)
    _finish(fig2)

    # ---------- 图3: 仰角限位检查(a: 段内相对时间, b: 绝对时间线) ----------
    t0 = Time(data["start_time"])
    fig3 = plt.figure(3, figsize=(7.2, 5.4))
    # fig3.suptitle("Elevation limit check within each scan", y=0.97)
    ax1 = fig3.add_subplot(211)
    ax2 = fig3.add_subplot(212)

    for i, (r, coord) in enumerate(zip(records, sky_coords)):
        dur = float(r["duration"])
        npts = int(dur) + 1
        t_start = Time(r["start_time_str"])
        time_pts = t_start + np.linspace(0.0, dur, npts) * u.minute
        _, el = src_az_el(coord, time_pts, site)
        minutes_rel = np.linspace(0.0, dur, npts)  # 积分内相对分钟
        hours_rel = (time_pts - t0).to(u.hour).value  # 相对调度起始时刻的小时数
        ax1.plot(minutes_rel, el, lw=0.8, color=_C_SEC, alpha=0.55)
        ax2.plot(hours_rel, el, lw=0.8, color=_C_SEC, alpha=0.55)

    ax1.set_xlabel("Time within scan (min)")
    ax1.set_ylabel("Elevation (deg)")
    ax2.set_xlabel(f"Hours from start (UTC {data['start_time']})")
    ax2.set_ylabel("Elevation (deg)")
    for ax in (ax1, ax2):
        _constraint_band(ax, el_lim, el_max)
        _light_grid(ax)
        ax.text(0.01, 0.98, f"n = {len(records)} targets",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=_C_NEU)
    _panel_label(ax1, "a")
    _panel_label(ax2, "b")
    _finish(fig3)

    # ---------- 图4: 不同源覆盖累计进度(hero 面板) ----------
    # 横轴: 相对调度起始的累计观测时长(小时); 纵轴: 截至该时刻已观测的不同源数.
    # JD 差单位为天, 乘 24 转为小时(轴标签与单位一致).
    t0 = Time(data["start_time"])
    t0_jd = float(t0.jd)  # type: ignore
    starts_h = [(float(Time(r["start_time_str"]).jd) - t0_jd) * 24.0 for r in records]  # type: ignore
    ends_h = [(float(Time(r["end_time_str"]).jd) - t0_jd) * 24.0 for r in records]  # type: ignore
    starts_h = np.asarray(starts_h)
    ends_h = np.asarray(ends_h)
    names = [r["target"] for r in records]
    coverage_x = np.empty(len(records) * 2 + 1)
    coverage_y = np.empty(len(records) * 2 + 1)
    coverage_x[0] = 0.0
    coverage_y[0] = 0.0
    seen = set()
    unique_count = 0
    for i, (s_h, e_h, name) in enumerate(zip(starts_h, ends_h, names)):
        # step 起点先水平延伸(纵坐标未更新), 终点再垂直跳变
        coverage_x[2 * i + 1] = s_h
        coverage_y[2 * i + 1] = unique_count
        if name not in seen:
            seen.add(name)
            unique_count += 1
        coverage_x[2 * i + 2] = e_h
        coverage_y[2 * i + 2] = unique_count

    fig4 = plt.figure(4, figsize=(7.2, 3.2))
    # fig4.suptitle("Unique-source coverage progress", y=0.98)
    ax_cov = fig4.add_subplot(111)
    ax_cov.step(coverage_x, coverage_y, where="post", lw=1.4, color=_C_MAIN, zorder=2)
    ax_cov.fill_between(coverage_x, coverage_y, step="post", color=_C_SEC, alpha=0.15, zorder=1)
    ax_cov.plot([coverage_x[-1]], [coverage_y[-1]], "o", ms=5, color=_C_RED, zorder=3)
    ax_cov.set_xlabel("Accumulated observing time (h)")
    ax_cov.set_ylabel("Covered unique sources")
    ax_cov.set_xlim(0.0, float(coverage_x[-1]) * 1.05)
    ax_cov.set_ylim(0.0, float(coverage_y[-1]) * 1.18)
    # 横轴跨度通常 24~72h, 12h 步长便于读出整夜进度
    x_max = float(coverage_x[-1]) * 1.05
    ax_cov.set_xticks(np.arange(0.0, x_max + 1e-9, 12.0))
    ax_cov.text(0.01, 0.98,
                f"n = {len(records)} segments \u2192 {coverage_y[-1]:.0f} unique targets",
                transform=ax_cov.transAxes, ha="left", va="top", fontsize=7, color=_C_NEU)
    _light_grid(ax_cov)
    _finish(fig4)

    # ---------- 图5: AZ-EL 起点的连续 strip ----------
    # 望远镜方位转动范围 ±270°(-90°~450°): 每个观测起点 az 沿最短转动路径展开,
    # 使折线连续且保持在 [-90, 450] 内; 线段上画方向箭头; 序号标注在起点右上方,
    # 偏移避免与原点重叠(从 1 开始, 53 个); 叠加仰角限位带.
    az_starts = []
    el_starts = []
    for r, coord in zip(records, sky_coords):
        t_start = Time(r["start_time_str"])
        az, el = src_az_el(coord, t_start, site)
        az_starts.append(float(az) % 360.0)
        el_starts.append(float(el))
    az_starts = np.asarray(az_starts)
    el_starts = np.asarray(el_starts)

    # 卷绕展开: 望远镜方位转动范围 ±270°(-90°~450°), 每个点的展开候选限定在
    # [-90, 450] 内(az 0~360 及其 ±360 等价角度中落在范围内的). 为避免贪心
    # 在边界附近"被困"后被迫大幅回转, 用动态规划选每个点的展开值, 使望远镜
    # 总行程(相邻差绝对值之和)全局最小.
    cands = [
        [a + 360.0 * k for k in (-1, 0, 1) if -90.0 <= a + 360.0 * k <= 450.0]
        for a in az_starts
    ]
    dp = [0.0] * len(cands[0])          # 到当前点选各候选的最小累计行程
    back = []                            # 每个点的前驱选择(用于回溯)
    for i in range(1, len(cands)):
        cur_dp, cur_back = [], []
        for cj in cands[i]:
            best_k = min(range(len(cands[i - 1])), key=lambda k: dp[k] + abs(cj - cands[i - 1][k]))
            cur_dp.append(dp[best_k] + abs(cj - cands[i - 1][best_k]))
            cur_back.append(best_k)
        dp, back = cur_dp, back + [cur_back]
    # 回溯最优路径
    j = int(np.argmin(dp))
    unwrapped_rev = [cands[-1][j]]
    for i in range(len(cands) - 1, 0, -1):
        j = back[i - 1][j]
        unwrapped_rev.append(cands[i - 1][j])
    unwrapped = np.asarray(unwrapped_rev[::-1])

    fig5 = plt.figure(5, figsize=(7.2, 3.6))
    ax_strip = fig5.add_subplot(111)
    _constraint_band(ax_strip, el_lim, el_max)
    ax_strip.set_xlabel("Unwrapped Azimuth (deg)")
    ax_strip.set_ylabel("Elevation (deg)")

    # 折线轨迹(每个线段加方向箭头)
    ax_strip.plot(unwrapped, el_starts, "-o", ms=4, lw=1, color=_C_MAIN, zorder=2)
    if len(unwrapped) >= 2:
        for i in range(len(unwrapped) - 1):
            ax_strip.annotate(
                "",
                xy=(unwrapped[i + 1], el_starts[i + 1]),
                xytext=(unwrapped[i], el_starts[i]),
                arrowprops=dict(arrowstyle="-|>", color=_C_MAIN, lw=0.8,
                                mutation_scale=8, shrinkA=2, shrinkB=2),
                zorder=3,
            )

    # 在每个起点右上方标注目标序号(从 1 开始), 偏移避免与原点重叠
    off_x = 0.5   # 方位角偏移(度), 约 X 轴跨度 540° 的 0.7%
    off_y = 0.5   # 仰角偏移(度)
    for i, (a, e) in enumerate(zip(unwrapped, el_starts), start=1):
        ax_strip.text(a + off_x, e + off_y, str(i), fontsize=6.5,
                      ha="left", va="bottom", color=_C_NEU_DARK, zorder=4)

    # ax_strip.text(0.01, 0.98, f"n = {len(records)} start points",
    #               transform=ax_strip.transAxes, ha="left", va="top", fontsize=7, color=_C_NEU)
    ax_strip.set_xlim(-90.0, 450.0)
    ax_strip.set_xticks(np.arange(-90, 451, 90))
    _light_grid(ax_strip)
    _finish(fig5)

    # ---------- 输出: 保存(SVG/PDF/TIFF/PNG) / 显示 ----------
    figs = [fig1, fig2, fig3, fig4, fig5]
    paths = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        names = [
            "fig1_slew_time",
            "fig2_start_end_el",
            "fig3_el_limit_check",
            "fig4_unique_source_coverage",
            "fig5_az_el_strip",
        ]
        for fig, name in zip(figs, names):
            base = os.path.join(save_dir, name)
            fig.savefig(base + ".pdf", bbox_inches="tight")
            fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
            paths.append(base + ".png")
            print(f"[plot] 已保存: {base} (.pdf/.png)")
    if show and "agg" not in plt.get_backend().lower():
        plt.show()
    elif not show:
        # 批处理模式: 保存后立即释放内存
        for fig in figs:
            plt.close(fig)
    return {"figs": figs, "paths": paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot telescope scheduling results")
    parser.add_argument("--schedule-results", default="schedule_results.json",
                        help="path to schedule results JSON (default: schedule_results.json)")
    parser.add_argument("--targets", default=None,
                        help="path to target table JSON with ra/dec (default: tests/targets.json)")
    parser.add_argument("--config", default=None,
                        help="path to site config YAML (default: yn40mss default.yaml)")
    parser.add_argument("--save-dir", default=None,
                        help="directory to save figures (fig1..fig5_* in pdf/png); skip saving if omitted")
    parser.add_argument("--no-show", action="store_true", help="do not pop up figures")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    args = parser.parse_args(argv)

    return plot_schedule_results(
        schedule_results_file=args.schedule_results,
        targets_file=args.targets,
        config_file=args.config,
        save_dir=args.save_dir,
        show=not args.no_show,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
