# -*- coding: utf-8 -*-
"""端到端调度测试(v2 管线): 必排全覆盖 + fill-gap 填闲 + 零重叠

快速模式(test_mode=True)平衡 CI 时长; 全量验证见 README 的 CLI 用法。
"""
import json
import os
import unittest

from yn40mss.algorithm import dqn
from yn40mss.core.utils import load_config
from yn40mss.config import DEFAULT_CONFIG_PATH

TARGETS_FILE = os.path.join(os.path.dirname(__file__), "targets.json")


class ScheduleTestCase(unittest.TestCase):

    def test_schedule_full_coverage_and_fill_gap(self):
        cfg = load_config(DEFAULT_CONFIG_PATH)
        targets = json.load(open(TARGETS_FILE, "r", encoding="utf-8"))

        recs = dqn.do_schedule(
            targets, cfg,
            cfg.common.start_time, cfg.common.end_time,
            cfg.common.start_az, cfg.common.start_el,
            schedule_results_file=os.path.join(os.path.dirname(__file__),
                                               "schedule_results.json"),
        )

        self.assertTrue(recs, "scheduler produced no records")

        unique = {r["target"] for r in recs}
        self.assertEqual(len(unique), len(targets),
                         f"mandatory coverage {len(unique)}/{len(targets)}")

        fill_extra = len(recs) - len(unique)
        self.assertGreaterEqual(fill_extra, 1,
                                "expected at least one fill-gap extra")

        ordered = sorted(recs, key=lambda r: r["start_time"])
        overlaps = sum(1 for a, b in zip(ordered, ordered[1:])
                       if b["start_time"] < a["end_time"] - 1e-6)
        self.assertEqual(overlaps, 0, f"{overlaps} overlapping pairs")

        total_obs = sum(r["duration"] for r in recs)
        print(f"\nrecords={len(recs)} unique={len(unique)} "
              f"fill_gap_extra={fill_extra} obs={total_obs:.1f} min "
              f"slew={sum(r['slew_time'] for r in recs):.2f} min")


if __name__ == "__main__":
    unittest.main()
