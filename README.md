# YesTiger 本地部署版

YesTiger 是一个面向偶像 / Anisong 现场应援规划的 Web Studio。用户上传歌曲后，系统会估计音乐结构，生成 call / mix / 地下芸动作时间线，并支持在网页里编辑、保存备注、导出 JSON / Markdown / 视频。

这份仓库是“本地一键部署版”。

## 一、先安装这些软件

请先安装这些“系统工具”：

- Python 3.10 或 3.11
- Git
- Git LFS
- FFmpeg

YesTiger 不会把 PyTorch、transformers 等 Python 依赖安装到系统 Python 里。`setup_windows.ps1` / `setup_linux.sh` 会在项目目录下创建自己的虚拟环境：

```text
.venv/
```

后续启动也会固定使用这个虚拟环境里的 Python。系统里只需要先有一个 Python 解释器，用来创建 `.venv`。

Windows 推荐安装方式：

```powershell
winget install Python.Python.3.10
winget install Git.Git
winget install GitHub.GitLFS
winget install Gyan.FFmpeg
```

安装完后，关闭当前终端，重新打开一个 PowerShell。

检查是否安装成功：

```powershell
python --version
git --version
git lfs version
ffmpeg -version
```

如果这些命令都有输出版本号，就可以继续。

## 二、下载并启动 YesTiger

Windows PowerShell：

```powershell
git lfs install
git clone https://github.com/Doloris-Misumi/Yetiger.git
cd Yetiger
git lfs pull
.\setup_windows.ps1
.\start_windows.ps1
```

如果 PowerShell 提示禁止运行脚本，改用：

```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup_windows.ps1
PowerShell -ExecutionPolicy Bypass -File .\start_windows.ps1
```

启动后打开浏览器：

```text
http://127.0.0.1:8765
```

健康检查地址：

```text
http://127.0.0.1:8765/api/health
```

看到 `"status": "ok"` 表示后端加载完成。

## 三、Linux / macOS

先安装 Python、Git LFS、FFmpeg，然后运行：

```bash
git lfs install
git clone https://github.com/Doloris-Misumi/Yetiger.git
cd Yetiger
git lfs pull
bash setup_linux.sh
bash start_linux.sh
```

然后打开：

```text
http://127.0.0.1:8765
```

## 四、第一次使用会发生什么

`setup_windows.ps1` / `setup_linux.sh` 会自动：

- 创建 `.venv` Python 虚拟环境
- 安装 CPU 版 PyTorch / torchaudio
- 安装 YesTiger 需要的 Python 依赖
- 创建运行目录 `webapp_runs/`

仓库里已经包含 YesTiger 自己训练好的 checkpoint：

```text
train/runs/muq_seed46_110songs_tail_merge/checkpoint.pt
```

第一次真正分析歌曲时，程序可能还会从 Hugging Face 下载 MuQ 大模型：

```text
OpenMuQ/MuQ-large-msd-iter
```

如果想提前下载，可以在 setup 之后运行：

Windows：

```powershell
.\.venv\Scripts\python.exe pre_download_models.py
```

Linux / macOS：

```bash
.venv/bin/python pre_download_models.py
```

## 五、常见问题

### 页面只有链接和原生按钮，没有居中布局 / 拖拽区 / canvas

这是前端静态资源没有加载成功的表现。请先确认已经拉到最新代码：

```powershell
git pull
git lfs pull
.\start_windows.ps1
```

然后在浏览器里按 `Ctrl + F5` 强制刷新。正常页面应该有居中的入口卡片，Studio 页面应该有拖放音频区域和右侧 canvas 预览。

### checkpoint.pt 很小或者模型加载失败

通常是 Git LFS 没有拉到真实模型文件。重新运行：

```powershell
git lfs pull
```

正常的 `checkpoint.pt` 大约 84MB。

### ffmpeg not found

说明 FFmpeg 没装好，或者不在 PATH 里。Windows 可以重新运行：

```powershell
winget install Gyan.FFmpeg
```

然后重开 PowerShell。

### 第一次分析很慢

正常。第一次会下载 MuQ 模型并建立缓存。

### 用户数据保存在哪里

所有上传、分析结果、自定义 mix、备注都会保存在：

```text
webapp_runs/
```

备份这个目录即可保留用户数据。

## 六、项目目录

```text
webapp/                 网页与 Python HTTP API
train/                  MuQ 推理代码、结构标注、checkpoint
support/                应援动作推荐规则
knowledge/              call / mix / action 知识库
call_audio/             动作音频预览素材
gei_video/              地下芸视频素材
config/                 本地配置示例
requirements-webapp.txt Python 依赖
setup_windows.ps1       Windows 初始化脚本
start_windows.ps1       Windows 启动脚本
setup_linux.sh          Linux/macOS 初始化脚本
start_linux.sh          Linux/macOS 启动脚本
Dockerfile              可选 Docker 部署入口
```

更详细的部署说明见：

```text
DEPLOYMENT_LOCAL_ZH.md
```
