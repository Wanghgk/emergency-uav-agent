# Emergency-UAV-Agent

基于 ESARBench 的无人机应急搜救仿真平台。无人机从官方起点飞往模拟报警区，采集 RGB-D、检测人员并估计三维位置。搜索结束后，网页可在正射底图上回放航迹，查看人员位置、对应照片和复核状态。

## 安装和部署

首次在另一台电脑使用，请从 [Windows 逐步部署指南](docs/deployment-windows.md) 开始。该指南包含从 GitHub 克隆、创建 Conda 环境、获取官方资源、运行仿真和打开网页的 CMD 命令。

完成首次部署后，在仓库根目录的 CMD 或 Anaconda Prompt 启动网页服务。

```cmd
conda activate emergency-uav-agent
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

打开 [搜救回放](http://127.0.0.1:8000/)。在另一个已激活同一 Conda 环境的 CMD 窗口从仓库根目录运行任务，生成本机观测记录。

```cmd
python -m scripts.run_rescue --task-id 290
```

默认使用本地规则做规划。需要用 DeepSeek 规划时，在 CMD 环境中配置 `DeepseekApikey`，并给运行命令加 `--deepseek`。

## 工程结构

| 路径 | 内容 |
| --- | --- |
| `backend/` | FastAPI、规划、人员识别、定位与仿真接口 |
| `frontend/` | React + JavaScript 回放界面，使用 Vite 构建 |
| `configs/` | 模拟报警范围与预计人数 |
| `scripts/` | 场景验收、人员搜救、正射采集与复查 |
| `tests/` | 坐标、规划、匹配与结果数据测试 |
| `.external/` | 本机下载的官方源码、场景与模型权重 |
| `data/` | 仿真生成的地图、RGB-D 与任务记录 |

GitHub 仓库提供源码、配置、测试和文档。场景、权重、观测、构建产物、环境与凭据存放在本机。

- [P0 环境安装及复现](docs/esar-p0.md)
- [P1 正射底图、人员检测和本地 LLM 搜索](docs/esar-p1.md)
- [三地图接警搜救、结果回放及复查](docs/esar-dispatch.md)

本项目依赖 [ESARBench](https://github.com/4amGodvzx/ESAR)、[AirSim-Colosseum](https://github.com/CodexLabsLLC/Colosseum) 和 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)。第三方资源保留其上游许可。
