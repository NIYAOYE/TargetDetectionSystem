# SAR 目标检测浏览器服务

这是一个基于浏览器和后端服务的 SAR 图像目标检测项目。项目提供静态 Web 界面、FastAPI 后端、模型与图像文件管理、检测任务执行、人工框选修正以及结果导出能力。

## 功能概览

- 在浏览器中预览 SAR 图像并发起目标检测任务。
- 支持上传和管理图像、YOLO 检测模型、随机森林分类模型。
- 支持内置 `demo-synthetic` 检测器，用于在没有真实模型时验证完整流程。
- 支持检测框结果查看、手动新增、删除与导出。
- 导出内容包括叠加标注图像和 UTF-8-sig 编码的 CSV 文件。

## 项目结构

```text
backend/                 后端服务、任务管理、图像读写与算法调用
backend/algorithms/      YOLO、随机森林、图像处理与特征提取逻辑
config/                  默认配置文件
scripts/                 环境检查与演示图像生成脚本
static/                  浏览器前端页面与样式脚本
storage/images/          待检测 SAR 图像
storage/models/          检测模型与分类模型
storage/outputs/         导出的标注图像与 CSV 结果
tests/                   自动化测试
```

## 环境要求

- 推荐使用 Python 3.11。
- 推荐使用 Conda 创建环境，也可以使用普通虚拟环境。
- 在 Linux / WSL 环境中，OpenCV 可能需要额外的系统库，例如 `libgl1` 和 `libglib2.0-0`。

主要 Python 依赖包括：

- FastAPI / Uvicorn
- NumPy / Pandas
- OpenCV / Rasterio
- PyTorch / TorchVision
- Ultralytics
- scikit-learn / joblib

完整依赖请查看 `requirements.txt` 或 `environment.conda.yml`。

## 使用 Conda 启动

在终端进入项目根目录后执行：

```bash
conda env update -n SarProject -f environment.conda.yml
conda activate SarProject
python scripts/check_environment.py
python scripts/create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Windows 命令行中也可以使用反斜杠形式运行脚本：

```bat
python scripts\check_environment.py
python scripts\create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

服务启动后，在浏览器打开：

```text
http://localhost:8000
```

## 使用虚拟环境启动

如果不使用 Conda，可以在项目根目录创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Windows PowerShell 激活虚拟环境的命令通常为：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 快捷启动脚本

如果已经准备好 `SarProject` Conda 环境，可以使用项目自带脚本启动服务。

Windows：

```bat
start_conda_windows.bat
```

Linux / WSL / macOS：

```bash
bash start_conda_ubuntu.sh
```

## 文件放置说明

启动服务前或运行过程中，可以将数据文件放入以下目录：

- `storage/images/`：SAR 图像，支持 `.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff`。
- `storage/models/detectors/`：YOLO 检测模型，支持 `.pt`、`.pth`。
- `storage/models/classifiers/`：随机森林分类模型，支持 `.pkl`、`.joblib`。
- `storage/outputs/`：检测结果导出目录，由服务自动写入。

也可以通过浏览器界面上传和删除图像、检测模型、分类模型。

## 配置说明

默认配置位于 `config/default_config.yaml`，主要包含：

- `yolo.device`：推理设备，默认为 `auto`。
- `yolo.conf_threshold`：检测置信度阈值。
- `yolo.iou_threshold`：YOLO 推理 IOU 阈值。
- `image_processing.patch_size`：切片大小。
- `image_processing.overlap`：切片重叠像素。
- `image_processing.nms_threshold`：后处理 NMS 阈值。
- `random_forest.enabled`：是否启用随机森林分类能力。
- `ui.default_rf_enabled`：前端默认是否勾选随机森林分类。

## API 简要说明

- `GET /`：打开浏览器界面。
- `GET /api/config`：读取前端所需配置。
- `GET /api/files`：列出服务端图像和模型文件。
- `POST /api/files/upload/{category}`：上传图像或模型文件。
- `DELETE /api/files/{category}/{filename}`：删除图像或模型文件。
- `GET /api/images/{name}/preview.png`：获取图像预览。
- `POST /api/jobs/detect`：创建检测任务。
- `GET /api/jobs/{job_id}`：查看任务状态与进度。
- `GET /api/jobs/{job_id}/detections`：查看检测结果。
- `POST /api/jobs/{job_id}/detections`：手动新增检测框。
- `DELETE /api/jobs/{job_id}/detections/{index}`：删除指定检测框。
- `POST /api/jobs/{job_id}/export`：导出标注图像和 CSV 文件。

其中 `{category}` 可选值包括：

- `images`
- `detectors`
- `classifiers`

## 开发与测试

运行测试：

```bash
pytest
```

检查环境：

```bash
python scripts/check_environment.py
```

生成演示 SAR 图像：

```bash
python scripts/create_demo_image.py
```

## 注意事项

- 真实 YOLO 模型和随机森林模型不会随项目自动生成，需要按需放入对应目录或通过页面上传。
- 没有真实检测模型时，可以选择内置的 `demo-synthetic` 检测器验证页面和接口流程。
- 导出文件会写入 `storage/outputs/`，可通过浏览器下载或在本地目录中查看。
- 请勿将包含敏感数据、私有模型或个人路径信息的文件提交到版本管理系统。
