# YesTiger 本地部署版

YesTiger 是一个面向偶像 / Anisong 现场应援规划的本地 Web Studio。它可以分析歌曲结构，生成 call / MIX / 地下芸动作时间线，并支持在网页里继续人工编辑、保存备注、加载本地视频、导出 JSON / Markdown / MP4 教学视频。

这份仓库是“用户本地一键部署版”。用户拉取代码后，在自己的电脑上运行，不需要云服务器。

## 主要功能

- 上传 `.mp3 / .wav / .flac / .m4a / .ogg` 歌曲并自动分析结构。
- 加载 10 首示例歌曲的分析结果。
- 浏览内置 MIX & Call Library，查看名称、喊词、小节长度、适用场景和风险。
- 编辑歌曲结构段落，时间点会按重拍 / 推断小节网格吸附。
- 动作时间线会跟随结构段落更新，并按动作时间和重拍重新计算小节数。
- 搜索和替换应援动作 / MIX。
- 添加自定义备注，备注时间同样支持吸附。
- MIX Builder 可以创建自定义 call / MIX 动作，并保存到本地运行目录。
- 右上角可以加载本地 MV / 现场视频，与歌曲音频统一播放和拖动。
- MP4 导出支持所见即所得：画面、右上角视频、原曲音频、call 音频、备注都会合成进去。
- 导出 JSON / Markdown / MP4。

## 一、先安装系统工具

请先安装：

- Python 3.10 或 3.11
- Git
- Git LFS
- FFmpeg

YesTiger 的 Python 依赖会安装到项目自己的虚拟环境，不会写进系统 Python：

```text
.venv/
```

Windows 推荐安装：

```powershell
winget install Python.Python.3.10
winget install Git.Git
winget install GitHub.GitLFS
winget install Gyan.FFmpeg
```

安装后关闭当前终端，重新打开 PowerShell，检查：

```powershell
python --version
git --version
git lfs version
ffmpeg -version
```

## 二、下载并启动

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

启动后打开：

```text
http://127.0.0.1:8765
```

健康检查：

```text
http://127.0.0.1:8765/api/health
```

看到 `"status": "ok"` 表示后端已经启动。

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

## 四、第一次 setup 会做什么

`setup_windows.ps1` / `setup_linux.sh` 会自动：

- 创建 `.venv` 虚拟环境。
- 安装 CPU 版 PyTorch / torchaudio。
- 安装 Web Studio 所需依赖。
- 创建 `webapp_runs/` 运行目录。

仓库里包含 YesTiger 当前使用的 checkpoint：

```text
train/runs/muq_seed46_110songs_tail_merge/checkpoint.pt
```

第一次真正分析歌曲时，程序可能还会从 Hugging Face 下载 MuQ 模型：

```text
OpenMuQ/MuQ-large-msd-iter
```

如果想提前下载：

Windows：

```powershell
.\.venv\Scripts\python.exe pre_download_models.py
```

Linux / macOS：

```bash
.venv/bin/python pre_download_models.py
```

## 五、网页使用流程

打开首页后可以进入：

```text
/studio.html    歌曲分析和教学视频编辑
/builder.html   自定义 MIX / Call 动作
/library.html   MIX & Call Library 参考页
/readme.html    网页内教程
```

Studio 常用流程：

1. 选择示例歌曲，或上传自己的音频。
2. 等待分析完成。
3. 在右侧结构编辑器里调整段落开始 / 结束时间和段落类型。
4. 在动作时间线里调整动作时间、动作类型和具体 call / MIX。
5. 添加备注。备注会显示在教学画面左上角。
6. 可选：加载本地 MV / 现场视频。视频会显示在右上角，并跟随歌曲播放。
7. 导出 JSON、Markdown 或 MP4。

MP4 导出需要 FFmpeg。导出的 MP4 会包含：

- 教学画面。
- 右上角加载的视频。
- 原曲音频。
- call_audio 中对应动作的 call / MIX 声音。
- 时间备注。

## 六、示例音频

仓库包含 10 首示例歌曲的分析 JSON，但不包含商业歌曲 MP3。请不要把没有授权的原曲音频提交到 GitHub。

如果你有合法来源的示例音频，可以放到：

```text
example_audio/
```

文件名使用示例歌曲 ID，例如：

```text
example_audio/athiscode.mp3
example_audio/brushupbrassup.mp3
example_audio/divinespell.mp3
example_audio/dokidokisingout.mp3
example_audio/futarikoto.mp3
example_audio/itsuaietara.mp3
example_audio/kokokarakokokara.mp3
example_audio/lemonsour.mp3
example_audio/shunkansummerday.mp3
example_audio/soundscape.mp3
```

添加后重启 YesTiger，再加载示例歌曲，播放器会使用对应本地音频。

## 七、用户数据保存在哪里

所有运行期数据都在：

```text
webapp_runs/
```

包括：

- 上传的歌曲音频。
- 分析结果。
- 人工编辑后的结构和动作时间线。
- 备注。
- 用户自定义 call / MIX。
- MP4 导出文件和导出调试 JSON。

备份 `webapp_runs/` 即可保留本地用户数据。这个目录不会提交到 GitHub。

## 八、项目目录

```text
webapp/                 网页和 Python HTTP API
webapp/static/          前端页面、样式和 Studio 逻辑
webapp/static/examples/ 10 首示例分析 JSON
train/                  MuQ 推理代码、结构标注和 checkpoint
support/                应援动作推荐规则
knowledge/              call / mix / action 知识库
call_audio/             call / MIX 声音素材
gei_video/              地下芸演示视频素材
example_audio/          可选本地示例原曲音频，不提交商业音频
config/                 本地配置示例
webapp_runs/            本地运行数据，不提交
requirements-webapp.txt Python 依赖
setup_windows.ps1       Windows 初始化脚本
start_windows.ps1       Windows 启动脚本
setup_linux.sh          Linux/macOS 初始化脚本
start_linux.sh          Linux/macOS 启动脚本
Dockerfile              可选 Docker 部署入口
```

## 九、常见问题

### 页面只有链接和原生按钮

这是前端静态资源没有加载成功，或浏览器缓存了旧文件。请先更新代码：

```powershell
git pull
git lfs pull
```

然后重启 YesTiger，并在浏览器按 `Ctrl + F5` 强制刷新。

### checkpoint.pt 很小或者模型加载失败

通常是 Git LFS 没拉到真实模型文件：

```powershell
git lfs pull
```

正常的 `checkpoint.pt` 大约 84MB。

### ffmpeg not found

说明 FFmpeg 没装好，或者不在 PATH：

```powershell
winget install Gyan.FFmpeg
```

安装后重开 PowerShell。

### MP4 导出没有 call 声音

请确认 `call_audio/` 已通过 Git LFS 拉取完整：

```powershell
git lfs pull
```

导出时会在 `webapp_runs/jobs/<job_id>/exports/` 生成 `*.export-debug.json`，里面会记录使用了哪些 call 音频和最终 ffmpeg 命令。

### 示例歌曲保存失败

请更新到最新版本。示例歌曲第一次保存时会自动在 `webapp_runs/jobs/example_<song_id>/` 下创建本地结果文件。

### 第一次分析很慢

正常。第一次会下载 MuQ 模型并建立缓存。

## 十、更新已有本地副本

如果已经 clone 过仓库，更新时运行：

```powershell
cd Yetiger
git pull
git lfs pull
PowerShell -ExecutionPolicy Bypass -File .\setup_windows.ps1
PowerShell -ExecutionPolicy Bypass -File .\start_windows.ps1
```

浏览器里按 `Ctrl + F5` 强制刷新。

更详细的本地部署说明见：

```text
DEPLOYMENT_LOCAL_ZH.md
```
