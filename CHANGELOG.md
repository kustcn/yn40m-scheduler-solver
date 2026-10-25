# Changelog

## [20261025-3] - 2026-10-25-3

- add independent schedule verifier: new 'verify' CLI subcommand and
  yn40mss.verify module (coverage / visibility windows recomputed from
  scratch / zero overlaps / record consistency / adjacent-pointing slew
  feasibility), structured report + process exit code for CI
- tests/test_verify.py: positive case on the real schedule plus three
  negative cases (missing target, overlapping records, out-of-window
  record) - 4 passed

## [20261025-2] - 2026-10-25-2

- plot: add fig6_sky_trajectory - telescope pointing trajectory on the
  local alt-az polar diagram (filled dots = mandatory observations,
  open rings = fill-gap extras, arrows = schedule order with a
  blue-to-red order colorbar, elevation limit band, start/end markers,
  coverage/utilization inset)

## [20261025] - 2026-10-25

- Scheduler core upgraded to the v2 pipeline: adaptive large neighborhood
  search (ALNS) with four destroy operators (random / worst-slew /
  time-related / sky-related), success-rate adaptive operator selection,
  threshold-annealing acceptance, top-3 seed portfolio (40% / 60% budget)
- Fill-gap reuse integrated into the ALNS recreate step (plus post-hoc
  safety net); all targets are mandatory, fill-gap eligible sources may
  be re-observed to absorb idle time
- Coverage accounting fixed to distinct-target basis (a source's mandatory
  slot can no longer be traded away by the search)
- `schedule` results JSON now carries v2 summary fields
  (unique_targets / fill_gap_extra / total_obs_minutes / ...) and absolute
  time strings per record; scheduling summary prints unique coverage and
  fill-gap extras separately
- Add end-to-end test `tests/test_schedule.py` (`make test`)

## [20260831-2] - 2026-08-31-2

- Add plot sub command
- README.md 

## [20260831] - 2026-08-31

- Initial project structure
- Command-line interface tools
