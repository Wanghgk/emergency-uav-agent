# P1 罗布泊任务与地形底图

## 底图和坐标

`scripts.capture_basemap` 使用 ESARBench 罗布泊任务 290 的官方 UE5 场景，加装一个北向上、朝下的外置正射相机。采集 1536×1536 像素，覆盖起始点周围 1200×1200 米。采集时检查相机姿态与 UE 返回的投影矩阵，保存 WebP 和 NED 范围所需的元数据。网页以东坐标映射横轴，以北坐标映射纵轴。实采航点、检测估计位置及人工点选位置共用同一映射，可切换全图和放大航迹。

默认数据路径是 `data/maps/lopnur-290`。当前覆盖任务区域，地图范围以 `basemap.json` 的相机中心和正射宽度为准。[Colosseum 相机设置](https://codexlabsllc.github.io/Colosseum/settings/)

恢复或重新生成底图时，在 CMD 执行以下命令。

```cmd
conda activate emergency-uav-agent
python -m scripts.capture_basemap
```

## 搜索任务

任务入口 `scripts.run_mission` 向大脑提供自然语言需求和候选搜索扇区。大脑返回 Pydantic 校验过的计划，小脑在本机 AirSim 依次飞到指定航点，并用 YOLOX-Tiny 实际处理前向 RGB。检测框中心区域有可靠深度时才输出估计地图坐标，深度离散程度用于标记定位质量。首次搜索反馈触发一次 LLM 重规划，选择不同的航线。规划输入仅使用公开需求、候选搜索点和实采观测。日志和图像按独立运行 ID 关联，网页通过 WebSocket 收到更新。

首次安装 P1 推理依赖和权重时执行以下 CMD 命令。

```cmd
conda activate emergency-uav-agent
python -m pip install onnxruntime==1.32.0
if not exist ".external\models" mkdir ".external\models"
curl.exe -fL "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx" -o ".external\models\yolox_tiny.onnx"
```

使用 DeepSeek 时，从已加载 `DeepseekApikey` 的新 CMD 窗口运行，默认模型为 `deepseek-flash`。规划请求包含任务文字、可选扇区和检测计数。图像、深度、检测框和坐标保存在本机。[DeepSeek API](https://api-docs.deepseek.com/)

```cmd
conda activate emergency-uav-agent
python -m scripts.run_mission --deepseek
```

要采集较长的演示，使用 `python -m scripts.run_mission --deepseek --extended`。每个扇区走两条往返搜索航线，沿途在间距不超过约 22 米的采样航点记录 RGB-D 和检测结果。完成后在页面选择对应的“长航线”记录。回放按采集时间推进，可选 1×、2× 或 4×。

使用已启动的本机 Ollama 后端时，模型名可换为本机可用的兼容模型。低显存设备同时运行 UE5 和 Ollama 时，建议将 Ollama 配为 CPU 推理或使用 DeepSeek。

```cmd
conda activate emergency-uav-agent
set "ESAR_LLM_BASE_URL=http://127.0.0.1:11434/v1"
set "ESAR_LLM_MODEL=llama3.2:latest"
set "ESAR_LLM_TIMEOUT=90"
set "ESAR_LLM_API_KEY="
python -m scripts.run_mission
```

`--instruction` 可覆盖官方任务中公开的自然语言描述。`--mock` 用于不启动 Ollama 的调试，视觉推理仍基于真实 ESAR 图像。网页启动命令见 [README](../README.md)。报告位于 `data/missions`，对应 RGB-D 存在 `data/observations`。

## 实测记录

DeepSeek Flash 在任务 290 选择东侧作为第一航段。首轮 YOLOX-Tiny 反馈零检出后，DeepSeek 选择西侧作为第二航段，无人机完成两段水平移动。实采起点、东端和西端共三帧，东西端水平间距约 93 米，均与罗布泊底图配准。

三帧人员检出均为 0，日志中的完成状态表示两段搜索已执行。人员正检出及其位置误差需要在观察到目标的后续仿真航段验证。实际报告为 `data/missions/0c8152365c3d4fbf935de2068bb8c4da.json`。

长航线实测报告为 `data/missions/1554b4396be947f7a10f703439c92d91.json`。DeepSeek 先选择东侧，读取观测反馈后改搜北侧。30 帧的首末观测相隔 146.03 秒，按实测采样位置计算的水平航迹长 547.06 米。YOLOX 记录了 8 个候选框，深度质量检查均未产出可靠定位。网页按实际时间显示为 2:26，浏览器验证通过。

## 人员渲染预检

罗布泊任务 290 和 280 分别使用 UE 的 `TaskNum=14` 和 `TaskNum=13`。两个任务各有五个官方人员坐标，转换到 NED 后都与对应的 UE Survivor Actor 在厘米以内一致。每处的两个近距离 RGB 视角均用于人工核对，能看到站立和倒地人员。任务 290 长航线的采样点距离五个目标最近约 149–176 米，30 帧中只有 7 帧的投影范围包含目标位置。投影落入画面仅表示几何可见范围，地形遮挡和像素尺寸需另行判断。

此预检通过调试相机和无人机前视相机采集 RGB-D，输出到 `.external/diagnostics/person-rendering`。它读取官方目标真值，适用于独立诊断或评估。正常任务的搜索、模型和网页读取各自原有的观测数据。

任务 290 的实际前视相机诊断帧中，YOLOX-Tiny 输出 3 个人员框，其中 1 个通过深度定位检查，估计水平坐标距离对应官方 Actor 约 0.23 米。其他 2 个框无可靠定位。这个结果验证近距离链路，单帧误差不代表巡航定位精度。

```cmd
conda activate emergency-uav-agent
python -m scripts.verify_person_rendering --task-id 290
```

切换任务时，在前一个 UE5 进程退出后重新运行。检查目录中的 `contact-sheet.jpg`、`report.json` 以及 `onboard` 下的前视图像。
