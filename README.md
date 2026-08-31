# YN40m Targets Scheduling based DQN

[![Static Badge](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
![Powered by YN40M](https://img.shields.io/badge/Powered%20by-YN40m-orange.svg?style=flat&amp;colorA=E1523D&amp;colorB=007D8A)

## 依赖

### 运行时依赖

- numpy==1.26.4
- astropy==6.0.0
- matplotlib==3.7.1
- click==8.3.0
- torch>=2.0.0
- astroplan==0.10
- tqdm==4.66.2

### 可选依赖组

- `test`: 测试相关依赖 (pytest, coverage 等)
- `build`: 构建相关依赖 (build, wheel)
- `deploy`: 部署相关依赖 (twine, bumpver)
- `dev`: 开发环境全套依赖 (包含上述所有)

## 安装

`yn40m-scheduler-solver` 可用如下方式安装

### 源码安装

```shell
git clone http://github.com/kust/yn40m-scheduler-solver.git
cd yn40m-scheduler-solver

# 基础安装
pip install .

# 开发模式安装 (包含所有开发依赖)
pip install -e ".[dev]"

# 或者只安装测试依赖
pip install -e ".[test]"
```

## 运行与使用

`yn40m-scheduler-solver` 通过 CLI 入口运行,安装后会自动注册 `yn40m-scheduler-solver` 命令。

### 查看帮助

```bash
yn40m-scheduler-solver --help
yn40m-scheduler-solver --version
```

### 命令行参数

| 选项 | 必填 | 说明 |
| --- | --- | --- |
| `-c`, `--config-file` | ✅ | 配置文件(YAML)，参考 [default.yaml](src/yn40mss/config/default.yaml) |
| `-t`, `--targets-file` |  ✅  | 需要编排的目标文件(JSON) |


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

## 变更日志

请参阅 [CHANGELOG.md](CHANGELOG.md) 了解项目的变更历史。
