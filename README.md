# 操作视频 → 步骤操作文档生成器

将操作视频自动转换为带截图、带文字的步骤操作文档（Markdown + PDF）。

默认采用**字幕驱动**模式：Whisper 提取语音生成带时间戳的字幕，AI 分析字幕文本识别操作步骤，成本低、速度快。也可选择上传视频给 AI 看画面的增强模式。

## 工作流程

1. **Whisper 生成字幕** — 调用本地 Whisper 从视频提取语音转文字（带时间戳）
2. **AI 分析字幕** — 将字幕文本发送给 doubao 模型，识别操作步骤及最佳截图时间点，并给出自信度评分
3. **生成截图** — 根据每个步骤的时间点，用 ffmpeg 从视频中截取画面
4. **AI 看图增强** — 对自信度较低的步骤（最多 4 个），发送截图给 AI 看画面修正描述
5. **生成操作文档** — AI 生成结构化的 Markdown 操作文档（可选联网搜索增强）
6. **生成 PDF** — 将 Markdown 转为 PDF，截图自动嵌入

所有产物（视频、字幕、截图、文档）统一输出到以视频名命名的文件夹中。

## 安装依赖

```bash
pip install -r requirements.txt
```

确保系统已安装 `ffmpeg` 和 `whisper`。

## 配置

在项目根目录创建 `.env` 文件：

```env
# 必填：火山引擎 ARK API Key
ARK_API_KEY=your_ark_api_key

# 可选：Whisper 模型（默认 base）
WHISPER_MODEL=base
```

## 使用方法

### 基本用法

```bash
python video_analyzer_agent.py --video_path your_video.mp4
```

### 使用已有字幕文件（跳过 Whisper）

```bash
python video_analyzer_agent.py --video_path your_video.mp4 --srt_path subtitles.srt
```

### 启用联网搜索增强（需要 API 支持）

```bash
python video_analyzer_agent.py --video_path your_video.mp4 --web_search
```

### 视频上传分析模式（AI 看画面，较贵）

```bash
python video_analyzer_agent.py --video_path your_video.mp4 --use_video
```

### 完整参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--video_path` | 视频文件路径 | 自动查找当前目录MP4 |
| `--srt_path` | SRT字幕文件路径 | 无（自动用Whisper生成） |
| `--output_dir` | 输出目录 | 以视频文件名命名 |
| `--fps` | 抽帧频率（帧/秒） | 1 |
| `--whisper_model` | Whisper模型 | base |
| `--use_video` | 启用视频上传分析模式 | 关闭 |
| `--file_id` | 已上传的视频文件ID | 无 |
| `--max_vision` | AI 看图增强最大次数 | 4 |
| `--web_search` | 启用联网搜索增强文档 | 关闭 |
| `--api_key` | ARK API Key | 从.env读取 |

## 输出示例

运行后会在输出文件夹中生成：

```
your_video/
  your_video.mp4          ← 原始视频
  your_video.srt          ← 字幕文件
  images/
    step_01.jpg ~ step_N.jpg  ← 每步截图
  steps.json              ← 步骤分析数据
  operation_guide.md      ← Markdown 操作文档
  operation_guide.pdf     ← PDF 操作文档（图片嵌入）
```

文档格式示例：

```markdown
# 操作指南：XXX

## 步骤 1：打开设置页面

![步骤1截图](images/step_01.jpg)

点击屏幕左上角的菜单图标，在弹出的侧边栏中选择「设置」选项。

## 步骤 2：修改配置

![步骤2截图](images/step_02.jpg)

在设置页面中找到「高级选项」，点击进入后修改相关配置项。
```

## 致谢

本项目基于 [tech-shrimp/video_2_markdown_doubao](https://github.com/tech-shrimp/video_2_markdown_doubao) 改造而来。原项目实现了视频转学习笔记的基本功能，本项目在此基础上进行了以下改进：

- 新增字幕驱动模式，大幅降低 API 调用成本
- 新增自信度评分 + AI 看图增强机制
- 新增 PDF 输出
- 新增联网搜索增强（可选）
- 统一输出文件夹管理
