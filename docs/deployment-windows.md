# Windows 部署指南

以下命令使用 **Windows CMD 或 Anaconda Prompt**。路径以仓库根目录为基准，可以克隆到任意工作目录。

## 1. 准备工具

需要 Windows x64、支持 DirectX 12 的显卡驱动、[Git for Windows](https://git-scm.com/download/win)、[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 和 [Node.js](https://nodejs.org/)（22.12 或更新的兼容版本）。官方 Windows 场景压缩包约 7.7 GB，下载和解压前建议预留至少 20 GB。官方场景已编译，可直接启动对应的 Windows 程序。[ESARBench 系统要求](https://github.com/4amGodvzx/ESAR#system-requirements)

```cmd
git --version
conda --version
node --version
npm --version
```

## 2. 克隆工程并创建 Python 环境

先用 CMD 切换到存放项目的目录，再执行以下命令。若 GitHub 要求登录，使用有该仓库读取权限的账号。

```cmd
git clone https://github.com/Wanghgk/emergency-uav-agent.git
cd emergency-uav-agent
conda env create --file environment.yml
conda activate emergency-uav-agent
python -m pip install -r requirements-simulation.txt
```

如果同名 Conda 环境已经存在，执行 `conda env update --name emergency-uav-agent --file environment.yml` 更新依赖。

## 3. 获取官方任务和 AirSim 客户端

固定到项目验证所用的两个上游提交。安装命令需在已激活的 Conda 环境执行。

```cmd
if not exist ".external" mkdir ".external"
git clone --filter=blob:none --no-checkout https://github.com/4amGodvzx/ESAR.git .external/ESAR-pinned
git -C .external/ESAR-pinned checkout --detach 0cc13d6f4128b7cf38cd2025a809759f641fb88b
git clone --filter=blob:none --sparse https://github.com/CodexLabsLLC/Colosseum.git .external/Colosseum
git -C .external/Colosseum checkout --detach c2703d0ccd15d5e72911461af0d064e671c8a29d
git -C .external/Colosseum sparse-checkout set PythonClient
python -m pip install --no-build-isolation --no-deps -e .external/Colosseum/PythonClient
python -m pip check
```

## 4. 下载 Windows 场景和 YOLOX 权重

首次可只解压罗布泊。已有同版本的地图包时，可以复制到 `.external` 并直接校验。下载异常时可重新运行同一条带 `-C -` 的命令续传，或从官方文件页下载到相同路径。 [官方场景包](https://huggingface.co/datasets/4amGodvzx/ESAR/blob/4d8ba28e66581815a4ef2e77694998244021f158/ESARMaps_win.zip)

```cmd
curl.exe -fL --retry 3 -C - -o ".external\ESARMaps_win.zip" "https://huggingface.co/datasets/4amGodvzx/ESAR/resolve/4d8ba28e66581815a4ef2e77694998244021f158/ESARMaps_win.zip"
certutil -hashfile ".external\ESARMaps_win.zip" SHA256
```

预期地图 SHA-256 为 `7eedb17be0bcc2794c4c92a15d69e0f24566374df0ad57244b39b3daba81d125`。确认一致后解压。

```cmd
tar -xf ".external\ESARMaps_win.zip" -C ".external" ESARMaps_win/lopnur
if not exist ".external\models" mkdir ".external\models"
curl.exe -fL --retry 3 -o ".external\models\yolox_tiny.onnx" "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
certutil -hashfile ".external\models\yolox_tiny.onnx" SHA256
```

预期模型 SHA-256 为 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`。需要鳌太线和 K2 时，从同一个已校验的 ZIP 解压其目录。

```cmd
tar -xf ".external\ESARMaps_win.zip" -C ".external" ESARMaps_win/aotai2800 ESARMaps_win/k2
```

## 5. 验收仿真并生成第一份回放

机队任务默认使用 YOLO11s-Pose 复核 RT-DETRv2 R18 的候选框。运行前在 CMD 下载两份固定版本的权重，并核对 SHA-256。可用 `--detector rtdetr` 比较未做姿态复核的结果，或用 `--detector yolox` 复现 YOLOX-Tiny。

```cmd
curl.exe -fL --retry 2 -o ".external\models\rtdetr_v2_r18vd.onnx" "https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX/resolve/1e28de7/onnx/model.onnx"
certutil -hashfile ".external\models\rtdetr_v2_r18vd.onnx" SHA256
curl.exe -fL --retry 2 -o ".external\models\yolo11s-pose.onnx" "https://huggingface.co/unileon-robotics/YOLO11-ONNX/resolve/307d2addf959c94d7caf7ed0e2048b54bf4cbea2/yolo11s-pose.onnx"
certutil -hashfile ".external\models\yolo11s-pose.onnx" SHA256
```

RT-DETRv2 预期 SHA-256 为 `583a236ac21c95a7fd94f284fc21485e42355bfef82c27011ba78fbc09ee87e2`，YOLO11s-Pose 为 `5dc8b39e042dc44130e196ec3f1bcd346edf249e05963c3d6c2b506d01870600`。如果下载地址连接超时，可把 `huggingface.co` 替换为 `hf-mirror.com` 并再次核对哈希。[RT-DETRv2 ONNX](https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX/blob/1e28de7/onnx/model.onnx) · [YOLO11s-Pose ONNX](https://huggingface.co/unileon-robotics/YOLO11-ONNX/blob/307d2addf959c94d7caf7ed0e2048b54bf4cbea2/yolo11s-pose.onnx)

```cmd
python -m pytest -q --basetemp .pytest-tmp
ruff check --no-cache backend scripts tests
python -m scripts.verify_esar --task-only
python -m scripts.verify_esar
```

在线验收会启动罗布泊、采集两帧并检查坐标关系。上一场 UE5 退出后再启动下一条仿真命令。继续运行接警搜救可同时生成地图底图、图像和任务报告，跳过下面的可选 DeepSeek 配置也可本地运行。

```cmd
python -m scripts.run_rescue --task-id 290 --max-frames 4
```

`--max-frames 4` 用于装机冒烟测试。完整网格和复查可省略该参数。地图、帧和报告分别写入 `data/maps`、`data/observations`、`data/missions`。

## 6. 构建并打开网页

从仓库根目录执行以下命令构建前端。

```cmd
cd frontend
npm ci
npm run build
cd ..
```

单开一个 CMD 或 Anaconda Prompt，进入仓库根目录，手动启动 FastAPI 后端。运行期间保留这个窗口；任务结束或在网页暂停后，按 `Ctrl+C` 停止后端。

```cmd
conda activate emergency-uav-agent
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)，选择任务 290 的回放。播放航迹、点击像素或查看底部的搜救结果。新电脑上的页面会显示该电脑采集或迁移过来的记录。

## 7. 可选配置 DeepSeek、运行完整任务

在 Windows 的用户环境变量中创建 `DeepseekApikey`，填入自己的 API 密钥，然后重新打开 CMD 或 Anaconda Prompt。密钥保存在本机环境变量中。运行前可用以下命令检查是否已加载，命令只显示状态。

```cmd
if defined DeepseekApikey (echo DeepSeek configured) else (echo DeepSeek not configured)
conda activate emergency-uav-agent
python -m scripts.run_rescue --task-id 290 --deepseek
```

鳌太线使用 `--task-id 28`，K2 白天和夜间分别使用 `--task-id 370`、`--task-id 372`。任务入口每次独占 AirSim 端口 41451，须顺序执行。[更多任务与复查命令](esar-dispatch.md)

## 常见问题

- **网页返回 404**。先在 `frontend` 运行 `npm ci` 和 `npm run build`，再从仓库根目录重启 FastAPI。
- **网页提示暂无观测**。先完成第 5 步。旧机器的运行记录可按原目录结构另行复制整个 `data`。需要显示对应的离线补检统计时同时复制 `.external/diagnostics` 下对应的 `*-rotation-recheck.json`。
- **找不到仿真程序**。检查 `.external\ESARMaps_win\lopnur\Windows\lopnur.exe`，并确认从仓库根目录执行 Python 命令。
- **AirSim 端口被占用或启动失败**。确认已有的 UE5 场景已退出，检查 DirectX 12 驱动和运行日志。如果缺少 Visual C++ 运行库，可安装 [Microsoft 官方 x64 Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)。
