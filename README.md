# YesTiger 本地部署版

YesTiger 是一个面向偶像 / Anisong 现场应援规划的 Web Studio。用户上传歌曲后，系统会估计音乐结构，生成 call / mix / 地下芸动作时间线，并支持在网页里编辑、保存备注、导出 JSON / Markdown / 视频。

这个目录是 GitHub 发布用的干净版本。用户只需要下载本目录，并按照 [DEPLOYMENT_LOCAL_ZH.md](DEPLOYMENT_LOCAL_ZH.md) 操作，就可以在自己的电脑上本地部署。

## 目录内容

```text
webapp/                 网页与 Python HTTP API
train/                  当前 MuQ 推理代码、结构标注、checkpoint
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

## 最快启动

Windows PowerShell：

```powershell
cd yetiger
.\setup_windows.ps1
.\start_windows.ps1
```

Linux / macOS：

```bash
cd yetiger
bash setup_linux.sh
bash start_linux.sh
```

然后打开：

```text
http://127.0.0.1:8765
```

如果浏览器能打开页面，再访问：

```text
http://127.0.0.1:8765/api/health
```

`status` 为 `ok` 表示后端和 analyzer 已经加载完成。第一次启动和第一次分析会比较慢，因为 MuQ 模型可能需要下载缓存。

## 注意

- 推荐 Python 3.10 或 3.11。
- 需要安装 Git LFS 才能正确拉取 `checkpoint.pt` 和媒体素材。
- Windows 用户需要安装 FFmpeg 并让 `ffmpeg` 出现在 PATH 中。
- 用户生成结果会写入 `webapp_runs/`，这个目录不会提交到 Git。
