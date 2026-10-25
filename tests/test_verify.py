# -*- coding: utf-8 -*-
"""verify_schedule 单元测试: 阳性(真实 v2 结果) + 阴性(篡改记录必须被抓出).

窗口重算较慢(约 1.5 min); 阴性用例用 --use-cache? 不, 独立重算一次后
monkeypatch 缓存? 简化: 阳性用例做全量验证(独立重算), 阴性用例复用
相同窗口结果——把 verify_schedule 拆两步太繁, 直接允许阴性用例也走
完整重算但用 pytest fixture scope=session 只跑一次完整验证, 阴性用例
通过篡改 base 数据 + 复用缓存 pkl(第一次跑后已写? verify 不写缓存)。

务实方案: 阳性全量一次; 阴性三条各自篡改 records 后调用 verify_schedule
并传 use_cached_windows=True(首次阳性跑完后调度目录已有 target_available_times.pkl,
时间范围匹配即可秒级复用)。CI 中 tests 先跑 test_schedule(生成 pkl)?顺序不定。
更稳: 阴性用例检查的项(coverage/overlap/consistency)不依赖窗口重算的精确值,
但 verify_schedule 总会重算。为控时长, 阴性统一 use_cached_windows=True 且
本地仓库已带 62h 的 pkl(调度验证时生成); 若 pkl 不匹配则退化为重算, 仍正确。
"""
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
