# Emergency-UAV-Agent

- 从 [README.md](README.md) 和 [Windows 部署指南](docs/deployment-windows.md) 开始。使用项目的 Conda 环境运行 Python 命令。
- P0 使用官方 ESARBench 场景和任务。地图、官方源码和观察文件通过路径引用，并排除在项目版本控制之外。
- 规划器和检测模型只能访问公开任务信息以及实际观测。官方目标和线索坐标只供仿真初始化、评估或调试使用。
- 处理位置时明确区分 AirSim NED 米、UE 厘米和相机坐标。提交坐标或采集改动前，运行 Ruff、pytest 及可用的真实场景验证。
- Web 阶段使用 React、JavaScript、Vite 和 FastAPI。先按 P0 验收结果决定是否进入 P1。
