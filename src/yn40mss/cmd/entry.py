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
from yn40mss.plot import plot_schedule_results


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main() -> None:
    """YN40m telescope scheduler solver."""


@main.command()
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
@click.option(
    "--results-file",
    "results_file",
    required=False,
    default="schedule_results.json",
    type=str,
    help="Path to save the schedule results JSON (consumed by the 'plot' subcommand).",
)
def schedule(
    config_file: str,
    targets_file: str,
    timeline: str = "telescope_schedule.png",
    results_file: str = "schedule_results.json",
) -> None:
    """Run DQN scheduling and save the results."""
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

        recs = dqn.do_schedule(targets, cfg, start_time, end_time, start_az, start_el,
                               schedule_results_file=results_file)
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

        logging.info(f"📄 Schedule results saved to: {results_file}")

        exit_code = 0
    except Exception:
        click.echo(f"{traceback.format_exc()}")
        exit_code = 1

    sys.exit(exit_code)


@main.command()
@click.option(
    "--schedule-results",
    "schedule_results",
    required=False,
    default="schedule_results.json",
    type=str,
    help="Path to the schedule results JSON.",
)
@click.option(
    "--targets",
    "targets_file",
    required=False,
    default=None,
    help="Path to the target table JSON with ra/dec "
         "(default: records' embedded ra/dec, else tests/targets.json).",
)
@click.option(
    "--config",
    "config_file",
    required=False,
    default=None,
    help="Path to the site config YAML (default: yn40mss default.yaml).",
)
@click.option(
    "--save-dir",
    "save_dir",
    required=False,
    default='./out',
    help="Directory to save the 3 PNGs; skipped if omitted.",
)
@click.option(
    "--no-show",
    "no_show",
    is_flag=True,
    default=True,
    help="Do not pop up the figures.",
)
@click.option(
    "--dpi",
    "dpi",
    required=False,
    default=300,
    type=int,
    help="PNG resolution (default: 300).",
)
def plot(
    schedule_results: str,
    targets_file: str,
    config_file: str,
    save_dir: str,
    no_show: bool,
    dpi: int,
) -> None:
    """Plot the schedule results (3 figures), optionally save as PNG."""
    try:
        plot_schedule_results(
            schedule_results_file=schedule_results,
            targets_file=targets_file,
            config_file=config_file,
            save_dir=save_dir,
            show=not no_show,
            dpi=dpi,
        )
        exit_code = 0
    except Exception:
        click.echo(f"{traceback.format_exc()}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
