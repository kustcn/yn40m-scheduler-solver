import copy
import json
import os
import unittest

from yn40mss.verify import verify_schedule

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "schedule_results.json")
TARGETS = os.path.join(HERE, "targets.json")


def _load():
    with open(RESULTS, "r", encoding="utf-8") as f:
        return json.load(f)


class VerifyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(RESULTS):
            raise unittest.SkipTest(
                f"{RESULTS} not found - run 'yn40m-scheduler-solver schedule' first")
        cls.base = _load()

    def test_positive_real_schedule_passes(self):
        report = verify_schedule(RESULTS, targets_file=TARGETS,
                                 use_cached_windows=True)
        self.assertTrue(report["ok"], msg=json.dumps(
            {k: v["errors"] for k, v in report["checks"].items()}, ensure_ascii=False))

    def _check_negative(self, mutate, expect_check):
        data = copy.deepcopy(self.base)
        mutate(data)
        tmp = os.path.join(HERE, "_verify_negative.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        try:
            report = verify_schedule(tmp, targets_file=TARGETS,
                                     use_cached_windows=True)
            self.assertFalse(report["ok"])
            self.assertFalse(report["checks"][expect_check]["passed"],
                             f"expected {expect_check} to fail; got: "
                             f"{ {k: v['passed'] for k, v in report['checks'].items()} }")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_negative_missing_target_fails_coverage(self):
        # 删掉某目标的所有记录 -> coverage FAIL
        def mutate(d):
            victim = d["records"][0]["target"]
            d["records"] = [r for r in d["records"] if r["target"] != victim]
        self._check_negative(mutate, "coverage")

    def test_negative_overlap_fails(self):
        # 把最后一条整体前移, 与前一条重叠 -> overlap FAIL
        def mutate(d):
            last = d["records"][-1]
            span = last["end_time"] - last["start_time"]
            prev = min(r["end_time"] for r in d["records"][:-1])
            last["start_time"] = prev - 1.0
            last["end_time"] = last["start_time"] + span
        self._check_negative(mutate, "overlap")

    def test_negative_out_of_window_fails(self):
        # 把某条唯一观测移到调度窗起点之前(不可能在任何可见窗) -> windows FAIL
        def mutate(d):
            r = d["records"][0]
            span = r["end_time"] - r["start_time"]
            r["start_time"] = -60.0
            r["end_time"] = -60.0 + span
        self._check_negative(mutate, "windows")


if __name__ == "__main__":
    unittest.main()
