# SAR Target Detection Browser Server

This is the browser/server migration of `sar_project_1`. The original PyQt project is left untouched.

## Recommended Conda Setup

The original desktop project uses the `SarProject` conda environment. This BS version can reuse that environment; PyQt may remain installed, but it is not used by the browser/server app.

Windows Anaconda Prompt / cmd:

```bat
cd /d D:\LProject\cdx\BS_SarProject
conda env update -n SarProject -f environment.conda.yml
conda activate SarProject
python scripts\check_environment.py
python scripts\create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Windows PowerShell, after `conda init powershell` has been configured:

```powershell
Set-Location D:\LProject\cdx\BS_SarProject
conda env update -n SarProject -f environment.conda.yml
conda activate SarProject
python scripts\check_environment.py
python scripts\create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

WSL2 Ubuntu:

```bash
cd /mnt/d/LProject/cdx/BS_SarProject
conda env update -n SarProject -f environment.conda.yml
conda activate SarProject
python scripts/check_environment.py
python scripts/create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open this URL in a Windows or Ubuntu browser:

```text
http://localhost:8000
```

If you prefer a helper script after the environment is ready:

```bat
start_conda_windows.bat
```

or:

```bash
bash start_conda_ubuntu.sh
```

## Optional Ubuntu / WSL2 venv Setup

From WSL2 Ubuntu:

```bash
cd /mnt/d/LProject/cdx/BS_SarProject
python3.11 --version
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/create_demo_image.py
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open this URL in a Windows or Ubuntu browser:

```text
http://localhost:8000
```

## Storage Layout

Put files here before refreshing the browser file list:

- `storage/models/detectors/` for YOLO `.pt` / `.pth` models
- `storage/models/classifiers/` for RF `.pkl` / `.joblib` models
- `storage/images/` for SAR `.png` / `.jpg` / `.jpeg` / `.tif` / `.tiff` images
- `storage/outputs/` for exported overlay images and CSV files

The file list always includes `demo-synthetic`, a built-in detector for checking the web workflow without a real YOLO model.

## API Summary

- `GET /` opens the browser UI
- `GET /api/files` lists server-side images and models
- `GET /api/images/{name}/preview.png` returns a browser preview
- `POST /api/jobs/detect` starts a detection job
- `GET /api/jobs/{job_id}` returns task progress
- `GET /api/jobs/{job_id}/detections` returns detection rows
- `POST /api/jobs/{job_id}/detections` adds a manual detection
- `DELETE /api/jobs/{job_id}/detections/{index}` deletes a detection
- `POST /api/jobs/{job_id}/export` writes overlay PNG and UTF-8-sig CSV
