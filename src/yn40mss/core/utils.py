import time
from tqdm import tqdm
import math
import numpy as np
from datetime import datetime, timedelta
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astroplan import Observer, FixedTarget
from pytz import timezone
from astropy.utils import iers
from astropy.utils.iers import conf
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

def load_config(cfg_path):
    config = OmegaConf.load(cfg_path)
    return config

def get_obs_site(cfg):
    site = EarthLocation(lat = cfg.common.lat * u.deg, lon = cfg.common.lon * u.deg, height = cfg.common.height * u.m)

    observer = Observer(name = cfg.common.observer_name,
                        location = site,
                        pressure = cfg.common.pressure * u.bar,
                        relative_humidity=0.11,
                        temperature= 0 * u.deg_C,
                        timezone = timezone(cfg.common.timezone),
                        description = cfg.common.observer_description)
    return site, observer

class Target(object):
    def __init__(self, name, ra, dec, obs_time, site) -> None:
        super(Target).__init__()
        self.name = name
        self.ra = ra
        self.dec = dec
        self.obs_time = obs_time  # observational time
        self.site = site
        self.sky_coord = SkyCoord(ra, dec)  # FixedTarget in astropy 
        self.fixed_target = FixedTarget(name=self.name, coord=self.sky_coord)  # FixedTarget in astroplan

    def get_az_el(self, current_time):
        src_altaz = self.sky_coord.transform_to(AltAz(obstime=current_time, location=self.site))
        az = src_altaz.az.degree
        el = src_altaz.alt.degree
        az, el = PointModel(az, el)
        return az, el
    
    def target_slew_time_to_this(self, target_src, at_time):
        az, el = target_src.get_az_el(at_time)
        return self.azel_slew_time_to_this(az, el, at_time)
    
    def azel_slew_time_to_this(self, az_src, el_src, at_time):
        """
        To calculate the time needed for telescope transition from (az_src, el_src) to self at 'at_time'

        Parameters
        ----------
        az_src: azimuth of the source, e.g. 0
            e.g. src1 = SkyCoord('04 37 16 -47 15 09', frame='icrs', unit=(u.hourangle, u.deg))
        el_src: elevation of the source, e.g. 90
        at_time: the UT time to start the transition

        Returns
        -------
        t_slew: the time of the transition, in minutes
        """
        v_az, v_el = 1.0, 0.6
        az_speed_up_degree = 2.449
        az_speed_down_degree = 3.531
        el_speed_up_degree = 1.460
        el_speed_down_degree = 0.879
        az_speed_up_time = 5.93
        az_speed_down_time = 14.95
        el_speed_up_time = 6.35
        el_speed_down_time = 10.87
        az_stable_degree = az_speed_up_degree + az_speed_down_degree
        az_stable_time = az_speed_up_time + az_speed_down_time
        el_stable_degree = el_speed_up_degree + el_speed_down_degree
        el_stable_time = el_speed_up_time + el_speed_down_time
        # t_stable = el_speed_down_time # for stablising at the beginning and the end
        az, el = self.get_az_el(at_time)
        azdif, eldif = az - az_src, el - el_src
        if abs(az - az_src) / v_az > abs(el - el_src) / v_el + el_stable_time - az_stable_time:
            t_stable = az_stable_time - az_stable_degree / v_az
        else:
            t_stable = el_stable_time - el_stable_degree / v_el

        if np.abs(azdif) <= az_speed_up_degree and np.abs(eldif) <= el_speed_up_degree:
            dtaz = np.abs(azdif) / az_speed_down_degree * az_speed_down_time
            dtel = np.abs(eldif) / el_speed_down_degree * el_speed_down_time
            return max(dtaz, dtel)
        else:
            niter = 5
            dt = 0
            # count = 0
            for k in range(niter):
                obstime = at_time + dt * u.second
                az, el = self.get_az_el(obstime)
                dt_buf = max(abs(az - az_src) / v_az, abs(el - el_src) / v_el)
                if abs(dt_buf - dt) < 0.002:
                    break
                else:
                    dt = dt_buf
                # count += 1
            return float(format(dt, '.5f')) + t_stable    
def PointModel(az, el):
    vpar = [-1.8535, -0.9027, -0.1154, 0.0354, -0.0703, -0.0069, -0.0074, -0.0231]
    az0 = np.deg2rad(az)
    el0 = np.deg2rad(el)

    delt_az0 = vpar[0] + vpar[3] * np.tan(el0) + vpar[4] / np.cos(el0) \
                + vpar[5] * np.cos(az0) * np.tan(el0) \
                + vpar[6] * np.sin(az0) * np.tan(el0)

    delt_el0 = vpar[1] + vpar[2] * np.cos(el0) - vpar[5] * np.sin(az0) \
                + vpar[6] * np.cos(az0) + vpar[7] / np.tan(el0)

    az1 = az + delt_az0
    el1 = el + delt_el0
    
    return az1, el1

def iers_init(cfg):
    iers.conf.auto_download = False
    iers.IERS_A_URL = cfg.common.IERS_data
    # iers_a = iers.IERS_A.open(iers.IERS_A_URL)
    iers.IERS.iers_table = iers.IERS_A.open(iers.IERS_A_URL)


def load_targets(targets_data, site) -> list[Target]:
    targets = []
    for t in targets_data:
        ra = t["ra"]
        dec = t["dec"]
        if len(t["ra"].split(" ")) == 3:
            ras = t["ra"].split(" ")
            ra = ras[0] + 'h' + ras[1] + 'm' + ras[2] + 's'
        if len(t["dec"].split(" ")) == 3:
            decs = t["dec"].split(" ")
            dec = decs[0] + 'd' + decs[1] + 'm' + decs[2] + 's'
        obs_time = float(t["obs_time"])
        target = Target(t["src"], ra, dec, obs_time, site)
        targets.append(target)
    return targets

def time_seconds(t):
    """
    将astropy.time转换为时间戳（秒为单位）。
    
    Args:
        t: 包含时间信息的对象，该对象的value属性是一个形如'YYYY-MM-DD HH:MM:SS.sss'的字符串。
    
    Returns:
        转换后的时间戳，单位为秒。
    
    """
    return time.mktime(time.strptime(t.value.split(".")[0], "%Y-%m-%d %H:%M:%S"))
#两个时间点之间的分钟数
def minutes(start_time, end_time):
    """
    计算两个astropy.time.Time类型时间对象之间的分钟数差。
    
    Args:
        start_time (astropy.time.Time): 起始时间对象
        end_time (astropy.time.Time): 结束时间对象
    
    Returns:
        int: 起始时间与结束时间之间的分钟数差
    
    """
    t00 = time.strptime(start_time.value, "%Y-%m-%d %H:%M:%S.%f")
    t11 = time.strptime(end_time.value, "%Y-%m-%d %H:%M:%S.%f")
    tt0 = time.mktime(t00)
    tt1 = time.mktime(t11)
    return int((tt1-tt0)//60)

def cal_avail_times(targets, observer, time_range, cfg):
    target_available_times = {}  # 可用的观测时间的开始时间和结束时间
    target_available_seconds = {}  # 可用的秒数
    for target in tqdm(targets, "calculating available times for targets"):
        available_times, available_seconds  = cal_avail_times_for_target(target, observer, time_range, cfg)
        target_available_times[target.name] = available_times
        target_available_seconds[target.name] = available_seconds
    return target_available_times, target_available_seconds

def cal_avail_times_for_target(target, observer, time_range, cfg):
    min_airmass, max_airmass = 1.0, 10.0
    available_times = []
    available_seconds = []

    total_minutes = minutes(time_range[0], time_range[1])

    time_ut = time_range[0] + np.linspace(0, total_minutes, total_minutes)*u.minute
    # print(f"computing {target.id}")
    airmass = observer.altaz(time_ut, target.fixed_target).secz

    idxs = np.argwhere((airmass >= min_airmass) & (airmass < max_airmass))

    idxs = idxs.flatten()

    # idxs = [i for i in idxs if src_el(target.sky_coord, time_ut[i]) > el_lim and src_el(target.sky_coord, time_ut[i]) < el_max]

    # 限位计算
    limit_idxs = []
    for i in idxs:
        _, el = target.get_az_el(time_ut[i])
        if el > cfg.common.elevating_lim and el < cfg.common.elevating_max:
            limit_idxs.append(i)

    last_idx = limit_idxs[0]
    start_idx = limit_idxs[0]
    end_idx = -1

    for i, idx in enumerate(limit_idxs):
        if idx - last_idx > 1:
            end_idx = last_idx

        if end_idx != -1 or i == len(limit_idxs) - 1:
            if i == len(limit_idxs) - 1:
                end_idx = limit_idxs[-1]
            available_times.append((time_ut[start_idx], time_ut[end_idx]))
            available_seconds.append(time_seconds(time_ut[end_idx]) - time_seconds(time_ut[start_idx]))

            end_idx = -1
            start_idx = idx

        last_idx = idx
    return available_times, available_seconds

def time_to_lst_hours(time, location):
    """
    Convert astropy.time.Time object to Local Sidereal Time (LST) in hours.

    Args:
        time (astropy.time.Time):  Time object.
        location (astropy.coordinates.EarthLocation):  Observer's location.

    Returns:
        float: Local Sidereal Time in hours.
    """
    conf.iers_degraded_accuracy = 'warn' # Set to 'warn' or 'silent' to handle times outside IERS range
    lst = time.sidereal_time('apparent', location)
    return lst.hour


def azel_to_radec(az, el, lat, lst_hours):
    """
    Convert Azimuth and Elevation coordinates to Right Ascension and Declination.

    Args:
        az (float): Azimuth angle in degrees (North=0, East=90, South=180, West=270).
        el (float): Elevation angle in degrees (horizon=0, zenith=90).
        lat (float): Observer's latitude in degrees (positive for Northern hemisphere).
        lst_hours (float): Local Sidereal Time in hours.

    Returns:
        tuple: (ra, dec) Right Ascension and Declination in degrees.
               ra is in the range [0, 360), dec is in the range [-90, 90].
    """
    az_rad = math.radians(az)
    el_rad = math.radians(el)
    lat_rad = math.radians(lat)
    lst_rad = math.radians(lst_hours * 15)  # Convert LST from hours to degrees

    dec_rad = math.asin(math.sin(lat_rad) * math.sin(el_rad) + math.cos(lat_rad) * math.cos(el_rad) * math.cos(az_rad))

    ha_rad = math.atan2(-math.cos(el_rad) * math.sin(az_rad),
                        math.cos(lat_rad) * math.sin(el_rad) - math.sin(lat_rad) * math.cos(el_rad) * math.cos(az_rad))

    ra_rad = lst_rad - ha_rad

    dec_deg = math.degrees(dec_rad)
    ra_deg = math.degrees(ra_rad)

    # Normalize RA to be in the range [0, 360)
    ra_deg = ra_deg % 360
    if ra_deg < 0:
        ra_deg += 360

    return ra_deg, dec_deg



def check_schedule_overlaps(schedule_results):
    """检查调度结果中是否有时间重叠, 返回重叠信息列表"""
    overlaps = []
    ordered = sorted(schedule_results, key=lambda o: o['start_time'])
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            obs1, obs2 = ordered[i], ordered[j]
            if obs2['start_time'] >= obs1['end_time']:
                break  # 已按开始时间排序, 后面不会再重叠
            overlap_start = max(obs1['start_time'], obs2['start_time'])
            overlap_end = min(obs1['end_time'], obs2['end_time'])
            if overlap_start < overlap_end:
                overlaps.append({
                    'target1': obs1.get('target', f'Target-{i}'),
                    'target2': obs2.get('target', f'Target-{j}'),
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_end - overlap_start
                })
    return overlaps


def check_unscheduled_targets(schedule_results, all_targets):
    """检查哪些目标没有被编排, 返回未编排目标列表"""
    scheduled_names = {obs.get('target', '') for obs in (schedule_results or [])}
    unscheduled_targets = []
    for target in all_targets:
        target_name = target.get('src', target.get('name', ''))
        if target_name not in scheduled_names:
            unscheduled_targets.append({
                'name': target_name,
                'ra': target.get('ra', 'Unknown'),
                'dec': target.get('dec', 'Unknown'),
                'obs_time': target.get('obs_time', 'Unknown')
            })
    return unscheduled_targets


def print_scheduling_summary(start_time, schedule_results, all_targets):
    """打印调度摘要信息"""
    print("\n" + "=" * 60)
    print("TELESCOPE SCHEDULING SUMMARY")
    print("=" * 60)

    # 解析开始时间, 用于将分钟偏移量转换为绝对时间字符串
    start_datetime = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    total_targets = len(all_targets)
    scheduled_targets = len(schedule_results) if schedule_results else 0

    print(f"Total targets: {total_targets}")
    print(f"Successfully scheduled: {scheduled_targets}")
    print(f"Scheduling success rate: {(scheduled_targets / total_targets) * 100:.1f}%")

    if schedule_results:
        total_obs_time = sum(obs['duration'] for obs in schedule_results)
        total_slew_time = sum(obs.get('slew_time', 0.0) for obs in schedule_results)
        print(f"Total observation time: {total_obs_time:.1f} minutes")
        print(f"Total slew time: {total_slew_time:.1f} minutes")

        print("\nSCHEDULED TARGETS:")
        print("-" * 40)
        for i, obs in enumerate(schedule_results, 1):
            # 将相对 start_time 的分钟偏移转换为绝对时间字符串
            obs_start = start_datetime + timedelta(minutes=obs['start_time'])
            obs_end = start_datetime + timedelta(minutes=obs['end_time'])
            print(f"{i:2d}. {obs.get('target', f'Target-{i}'):<15} | "
                  f"{obs_start.strftime('%Y-%m-%d %H:%M:%S')} - "
                  f"{obs_end.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Duration: {obs['duration']:.1f}min | Slew: {obs.get('slew_time', 0):.1f}min")

    unscheduled = check_unscheduled_targets(schedule_results, all_targets)
    if unscheduled:
        print(f"\nUNSCHEDULED TARGETS ({len(unscheduled)} targets):")
        print("-" * 50)
        for i, target in enumerate(unscheduled, 1):
            print(f"{i:2d}. {target['name']:<15} | RA: {target['ra']:<12} | "
                  f"DEC: {target['dec']:<12} | ObsTime: {target['obs_time']}min")
    else:
        print("\n🎉 ALL TARGETS SUCCESSFULLY SCHEDULED!")

    print("=" * 60)


def plot_schedule_timeline(schedule_results, start_time_str, end_time_str, save_path="schedule_timeline.png"):
    """
    绘制望远镜观测调度时间线图

    Args:
        schedule_results: 调度结果列表，每个元素包含target, start_time, end_time等信息
        start_time_str: 开始时间字符串
        end_time_str: 结束时间字符串
        save_path: 图像保存路径
    """
    if not schedule_results:
        print("No scheduling results to plot")
        return

    start_datetime = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

    n = len(schedule_results)
    fig, ax = plt.subplots(figsize=(16, max(8.0, 0.28 * n + 2.0)))
    colors = plt.cm.tab20(np.linspace(0, 1, n))  # type: ignore

    y_positions, labels = [], []
    for i, obs in enumerate(schedule_results):
        ax.barh(i, obs['duration'], left=obs['start_time'], height=0.8,
                color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
        y_positions.append(i)
        labels.append(f"{obs.get('target', f'Target-{i}')} ({obs['duration']:.0f}m)")
        mid_time = (obs['start_time'] + obs['end_time']) / 2
        obs_start = start_datetime + timedelta(minutes=obs['start_time'])
        obs_end = start_datetime + timedelta(minutes=obs['end_time'])
        ax.text(mid_time, i, f"{obs_start.strftime('%H:%M')}-{obs_end.strftime('%H:%M')}",
                ha='center', va='center', fontsize=6.5, weight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel('Observation Targets', fontsize=12, weight='bold')

    total_minutes = (end_datetime - start_datetime).total_seconds() / 60
    ax.set_xlim(0, total_minutes)
    ax.set_xlabel('Time (minutes from start)', fontsize=12, weight='bold')
    time_ticks = np.arange(0, total_minutes, 120)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([(start_datetime + timedelta(minutes=t)).strftime('%m-%d %H:%M')
                        for t in time_ticks], rotation=45)

    overlaps = check_schedule_overlaps(schedule_results)
    if overlaps:
        ax.text(0.02, 0.98, f"Warning: {len(overlaps)} overlaps detected!",
                transform=ax.transAxes, fontsize=12, color='red',
                weight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        for overlap in overlaps:
            ax.axvspan(overlap['overlap_start'], overlap['overlap_end'],
                       alpha=0.3, color='red', zorder=0)

    ax.set_title('Telescope Observation Schedule Timeline', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    total_obs_time = sum(obs['duration'] for obs in schedule_results)
    efficiency = (total_obs_time / total_minutes) * 100
    stats_text = f"Targets: {len(schedule_results)}\nTotal Obs Time: {total_obs_time:.1f}min\nEfficiency: {efficiency:.1f}%"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()