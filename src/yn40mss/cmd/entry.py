import json
import logging
import sys
import traceback

import click

from yn40mss import __version__
from yn40mss.algorithm import dqn
from yn40mss.config import DEFAULT_CONFIG_PATH
from yn40mss.core.logging import setup_logging
from yn40mss.core.utils import load_config, print_scheduling_summary, check_schedule_overlaps, plot_schedule_timeline

@click.command()
@click.version_option(version=__version__)

@click.option(
    "-c",
    "--config-file",
    "config_file",
    required=False,
    help="config file",
)
@click.option(
    "-t",
    "--targets-file",
    "targets_file",
    required=True,
    help="targets json file.",
)
@click.option(
    "--timeline",
    "timeline",
    required=False,
    default="telescope_schedule.png",
    type=str,
    help="Path to save the schedule timeline image.",
)

def main(
    config_file: str,
    targets_file: str,
    timeline: str = "telescope_schedule.png",
) -> None:
    try:
        setup_logging()
        logging.info(
            f"Running on config: {config_file}, targets: {targets_file}"
        )
        if config_file is None:
            config_file = DEFAULT_CONFIG_PATH
        
        cfg = load_config(config_file)

        start_time, end_time, start_az, start_el = cfg.common.start_time, cfg.common.end_time, cfg.common.start_az, cfg.common.start_el
        targets = json.load(open(targets_file, "r", encoding="utf-8"))
    
        recs = dqn.do_schedule(targets, cfg, start_time, end_time, start_az, start_el)
        print('dqn results:', recs)

        # 获取调度时间范围
        print_scheduling_summary(start_time, recs, targets)

        overlaps = check_schedule_overlaps(recs)
        if overlaps:
            logging.warning(
                f"Found {len(overlaps)} overlapping observations: "
                f"{[(o['target1'], o['target2']) for o in overlaps[:5]]}"
            )

        # 绘制调度结果时间线图
        if timeline:
            plot_schedule_timeline(recs, start_time, end_time, timeline)
            logging.info(f"\n📊 Visualization saved to: {timeline}")

        exit_code = 0
    except Exception:
        click.echo(f"{traceback.format_exc()}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
