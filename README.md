# Yunnan 40 m Radio Telescope Targets Scheduling based DQN

[![Static Badge](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
![Powered by YN40M](https://img.shields.io/badge/Powered%20by-YN40m-orange.svg?style=flat&amp;colorA=E1523D&amp;colorB=007D8A)

## Dependencies

### Runtime dependencies

- numpy==1.26.4
- astropy==6.0.0
- matplotlib==3.7.1
- click==8.3.0
- torch>=2.0.0
- astroplan==0.10
- tqdm==4.66.2

### Optional dependency groups

- `test`: testing-related dependencies (pytest, coverage, etc.)
- `build`: build-related dependencies (build, wheel)
- `deploy`: deployment-related dependencies (twine, bumpver)
- `dev`: full development environment dependencies (includes all of the above)

## Installation

`yn40m-scheduler-solver` can be installed in the following ways:

### Install from source

```shell
git clone http://github.com/kust/yn40m-scheduler-solver.git
cd yn40m-scheduler-solver

# Basic installation
pip install .

# Development mode installation (includes all development dependencies)
pip install -e ".[dev]"

# Or install test dependencies only
pip install -e ".[test]"
```

## Usage

`yn40m-scheduler-solver` is driven through a CLI entry point; after installation the `yn40m-scheduler-solver` command is registered automatically.

### View help

```bash
yn40m-scheduler-solver --help
yn40m-scheduler-solver --version
```

### Command-line options

### do scheduling

  yn40m-scheduler-solver schedule -c config.yaml -t targets.json

| Option | Required | Description |
| --- | --- | --- |
| `-c`, `--config-file` | ✅ | Configuration file (YAML); see [default.yaml](src/yn40mss/config/default.yaml) |
| `-t`, `--targets-file` | ✅ | Target list to be scheduled (JSON) |

### scheduling result plotting
  
  yn40m-scheduler-solver plot -- schedule-results --config config.yaml --targets targets.json --save-dir ./out 

| Option | Required | Description |
| --- | --- | --- |
| `--schedule-results` | ✅ | Schedule results (JSON) |
| `--config` | ✅ | Configuration file (YAML); see [default.yaml](src/yn40mss/config/default.yaml) |
| `--targets` | ✅ | Target list to be scheduled (JSON) |
| `--save-dir` | ✅ | Directory to save the plot |

## Citation
```
@article{wei2026dqn,
   author = {Wei, Shoulin and Heng Zhang and Hao, Longfei and Liang, Bo and Dai, Wei},
   title = {Deep Q-Network Scheduling Achieves Complete Short-Term Target Coverage on the Kunming 40 m Radio Telescope},
   journal = {In Preparation},
   year = {2026},
   type = {Journal Article}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the project's change history.
