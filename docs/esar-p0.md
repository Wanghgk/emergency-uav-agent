# ESARBench P0 环境与数据链路

## 版本和安装

- ESARBench 代码固定在 [0cc13d6](https://github.com/4amGodvzx/ESAR/tree/0cc13d6f4128b7cf38cd2025a809759f641fb88b)。
- AirSim Python 客户端使用官方说明推荐的 Colosseum，固定在 [c2703d0](https://github.com/CodexLabsLLC/Colosseum/tree/c2703d0ccd15d5e72911461af0d064e671c8a29d/PythonClient)。ESAR 官方 Windows 快速入门使用 Python 3.9。本工程验证的是独立 Conda Python 3.11 的 P0 依赖组合。[ESARBench 安装说明](https://github.com/4amGodvzx/ESAR#quick-start)
- [官方 Windows 地图包](https://huggingface.co/datasets/4amGodvzx/ESAR/blob/4d8ba28e66581815a4ef2e77694998244021f158/ESARMaps_win.zip) 为 7,700,025,897 字节，SHA-256 为 `7eedb17be0bcc2794c4c92a15d69e0f24566374df0ad57244b39b3daba81d125`。Windows 渲染要求支持 DirectX 12 的驱动。

首次安装请按 [Windows 部署指南](deployment-windows.md) 获取依赖和官方场景。可用 `--esar` 和 `--maps` 指向已有源码及地图目录。

## 任务与观测

验证脚本默认从官方 `task_data.json` 选择任务 290。场景根据官方 Windows 入口的约定，使用 `snapshot_id % 15` 作为 UE5 场景参数。加载器返回公开任务字段与环境初始化所需的起始位置。目标、线索和火源真值留在官方数据文件内。官方视觉检测的 YOLO-World 适配代码需要独立权重，它将用于 P1 选型。[官方启动脚本](https://github.com/4amGodvzx/ESAR/blob/0cc13d6f4128b7cf38cd2025a809759f641fb88b/source/main_platform_win.py) · [官方检测实现](https://github.com/4amGodvzx/ESAR/blob/0cc13d6f4128b7cf38cd2025a809759f641fb88b/baselines_adaption/base/detection.py)

同一 AirSim RPC 请求读取前视相机 RGB 与 `DepthPlanar`。以各自帧尺寸建立针孔内参，并分别记录两路相机位姿和采集时间戳。深度以沿光轴的米数保存，像素用相机四元数旋转到 NED 世界系，再按官方地图原点转换为 UE 坐标。针孔焦距由水平视场角推导，假设方形像素和居中主点。现场校验用短距离升降后的静态地形深度比较来检查两处位姿的一致性。此方法给出场景内相对一致性，绝对定位精度需用已知地标另行评估。[AirSim 图像 API](https://microsoft.github.io/AirSim/image_apis/) · [ESAR 坐标工具](https://github.com/4amGodvzx/ESAR/blob/0cc13d6f4128b7cf38cd2025a809759f641fb88b/source/coordinate_utils.py)

每帧在 `data/observations/<frame_id>/` 下保存 `observation.json`、`rgb.png` 和 `depth.npy`。验证脚本打印观测文件路径和选取的深度像素坐标。重放同一像素时，在下面的命令中替换 `FRAME_ID`、`U` 和 `V`。

```cmd
python -m scripts.verify_esar --replay "data\observations\FRAME_ID\observation.json" --pixel U V
```

无效或缺失深度会返回错误。报告的世界坐标来自该帧的深度和相机位姿。离线重放适合定位调试和固定观测实验，改变航迹需要启动在线仿真。

## P0 验收记录

Windows 本地验收使用 RTX 3060 Laptop GPU、Python 3.11.18 和 Colosseum AirSim Python 客户端 1.8.1。官方 `lopnur` 任务 290 的起始 NED 坐标为 `(1144.1, 1332.5, -16.0)` 米。场景启动、RPC 连接、初始化、飞行控制和双帧采集均通过。

| 检查项 | 实测结果 |
|---|---|
| 基础飞行动作 | 实际垂直升高 3.715 m，X/Y 未变化 |
| 前视采集 | 两帧，RGB 512×512，平面深度 128×128 |
| 跨位姿地形深度一致性 | 377 个重叠采样点，残差中位数 0.375 m |
| 离线验证 | 两帧保存并重读后，样本像素映射结果一致 |
| 自动化检查 | pytest 11 项通过，Ruff 通过 |

验收记录位于 `data/observations/03819be2d0a84c0a8342eebb260dc68b/observation.json` 和 `data/observations/cfc1872a8e2a49658501335d6868a367/observation.json`。第一帧可用 `--pixel 64 68` 重放。第二帧可用 `--pixel 64 69` 重放。

首轮运行发现传送后的第一帧可能保留旧相机位置，以及天空深度饱和为 65504 m。采集现会检查相机与无人机的位置关系并等待刷新，反投影也会拒绝该饱和值。P0 实跑范围为单场景、前视相机和短距离垂直航段。GPS、IMU、其他地图和绝对地标定位精度有待单独验证。

使用本机 Conda 命名环境 `emergency-uav-agent` 复测时，真实场景仍通过。无人机实际升高 4.865 m，两帧有 358 个重叠采样点，跨位姿深度残差中位数为 0.375 m。网页回放通过 Chrome 验证，覆盖真实图像加载、播放换帧、有效像素的地图坐标和无效深度提示。
