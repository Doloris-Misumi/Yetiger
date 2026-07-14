# YesTiger 本地部署文档

这份文档面向从 GitHub 下载 `yetiger/` 发布目录的用户。目标是让用户在自己的电脑上运行完整网页和后端推理服务。

## 1. 硬件建议

最低建议：

```text
CPU: 4 核以上
内存: 8GB 以上
磁盘: 至少 8GB 空余
Python: 3.10 或 3.11
```

更舒服的配置：

```text
CPU: 6 核以上
内存: 16GB 以上
磁盘: 20GB 空余
```

YesTiger 当前默认使用 CPU 版 PyTorch。第一次分析完整歌曲可能较慢，这是正常现象。

## 2. 下载代码

推荐使用 Git + Git LFS：

```bash
git lfs install
git clone https://github.com/<你的账号>/<你的仓库>.git
cd <你的仓库>/yetiger
git lfs pull
```

确认模型文件不是 LFS 指针：

```bash
ls -lh train/runs/muq_seed46_110songs_tail_merge/checkpoint.pt
```

它应该约 84MB。如果只有几百字节，说明 LFS 没有拉下来，需要重新执行：

```bash
git lfs pull
```

## 3. 安装 FFmpeg

Windows 推荐：

```powershell
winget install Gyan.FFmpeg
```

安装后重新打开终端，确认：

```powershell
ffmpeg -version
```

Ubuntu / Debian：

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1
```

macOS：

```bash
brew install ffmpeg libsndfile
```

## 4. Windows 部署

在 PowerShell 中进入 `yetiger/` 目录：

```powershell
cd path\to\repo\yetiger
```

首次初始化：

```powershell
.\setup_windows.ps1
```

启动服务：

```powershell
.\start_windows.ps1
```

如果 PowerShell 禁止运行脚本，可以使用：

```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup_windows.ps1
PowerShell -ExecutionPolicy Bypass -File .\start_windows.ps1
```

打开浏览器：

```text
http://127.0.0.1:8765
```

## 5. Linux / macOS 部署

进入 `yetiger/` 目录：

```bash
cd path/to/repo/yetiger
```

首次初始化：

```bash
bash setup_linux.sh
```

启动服务：

```bash
bash start_linux.sh
```

打开浏览器：

```text
http://127.0.0.1:8765
```

## 6. 预下载 MuQ 模型

第一次分析时，程序会从 Hugging Face 下载 `OpenMuQ/MuQ-large-msd-iter`。如果想提前下载：

Windows：

```powershell
.\.venv\Scripts\python.exe pre_download_models.py
```

Linux / macOS：

```bash
.venv/bin/python pre_download_models.py
```

模型缓存默认在：

```text
.hf/
```

## 7. 验证

启动后先访问：

```text
http://127.0.0.1:8765/api/health
```

常见状态：

```text
loading  服务已启动，analyzer 正在加载
ok       analyzer 已加载，可以使用
error    依赖或模型加载失败，需要看终端报错
```

再打开：

```text
http://127.0.0.1:8765/studio.html
```

推荐测试顺序：

```text
1. 打开内置示例
2. 打开动作库
3. 创建一个自定义 mix
4. 上传 30-60 秒音频试跑
5. 上传完整歌曲
```

## 8. 局域网访问

如果想让同一 Wi-Fi 下的手机或另一台电脑访问：

Windows：

```powershell
$env:YESTIGER_HOST="0.0.0.0"
.\start_windows.ps1
```

Linux / macOS：

```bash
YESTIGER_HOST=0.0.0.0 bash start_linux.sh
```

然后在局域网设备访问：

```text
http://<这台电脑的局域网IP>:8765
```

## 9. 数据保存位置

运行中生成的数据会保存在：

```text
webapp_runs/
```

其中包括：

```text
uploads/         用户上传音频
jobs/            分析结果、导出文件、备注
feature_cache/   MuQ 特征缓存
custom_actions/  用户自定义 mix / action
```

如果要备份用户数据，备份整个 `webapp_runs/` 即可。

## 10. 可选 Docker 部署

如果用户安装了 Docker，可以直接：

```bash
docker build -t yetiger-local .
docker run --rm -p 8765:7860 -v "$(pwd)/webapp_runs:/app/webapp_runs" yetiger-local
```

Windows PowerShell：

```powershell
docker build -t yetiger-local .
docker run --rm -p 8765:7860 -v "${PWD}\webapp_runs:/app/webapp_runs" yetiger-local
```

然后访问：

```text
http://127.0.0.1:8765
```

## 11. 常见问题

### checkpoint.pt 很小

说明 Git LFS 没有拉到真实模型文件。执行：

```bash
git lfs pull
```

### ffmpeg not found

安装 FFmpeg，并重新打开终端。

### 第一次分析很慢

正常。第一次需要下载 MuQ 模型并建立缓存。

### /api/health 显示 error

看启动终端里的 Python 报错。最常见原因是依赖安装失败、PyTorch 版本不兼容、模型文件没有拉到。

### 只想重装依赖

可以删除 `.venv/` 后重新执行 setup 脚本。不要删除 `webapp_runs/`，除非你确认不需要用户结果。
