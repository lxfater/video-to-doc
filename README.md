# 视频分析AI Agent

使用火山引擎的doubao-seed-1.8模型直接分析视频，提取关键图片位置。

## 安装依赖

```bash
pip install volcengine-python-sdk[ark] python-dotenv requests
```

## 配置API Key

### 方法一：使用.env文件（推荐）

1. 复制.env.example文件为.env
```bash
copy .env.example .env
```

2. 编辑.env文件，填入你的ARK API Key
```
ARK_API_KEY=your_actual_api_key_here
```

### 方法二：命令行参数

在运行时通过--api_key参数传入

## 使用方法

### 命令行方式（自动查找视频）

```bash
# 自动查找当前目录第一个MP4文件，使用默认抽帧频率0.3帧/秒
python video_analyzer_agent.py
```

### 命令行方式（指定视频文件和抽帧频率）

```bash
# 指定视频文件和抽帧频率（例如1帧/秒）
python video_analyzer_agent.py --video_path your_video.mp4 --fps 1
```

### 命令行方式（直接传入API Key）

```bash
python video_analyzer_agent.py --api_key YOUR_ARK_API_KEY --video_path your_video.mp4
```

### 代码调用（使用.env文件）

```python
from video_analyzer_agent import VideoAnalyzerAgent

# 创建Agent（自动从.env文件读取API Key）
agent = VideoAnalyzerAgent()

# 分析视频
results = agent.analyze_video("your_video.mp4", fps=0.5)

# 保存结果
agent.save_results(results, "results.json")
```

### 代码调用（直接传入API Key）

```python
from video_analyzer_agent import VideoAnalyzerAgent

# 创建Agent
agent = VideoAnalyzerAgent(api_key="YOUR_ARK_API_KEY")

# 分析视频
results = agent.analyze_video("your_video.mp4")

# 保存结果
agent.save_results(results, "results.json")
```

## 输出格式

```json
[
    {
        "time": "01:23",
        "reason": "这张图片展示了关键操作，信息量很大"
    },
    {
        "time": "02:15",
        "reason": "这张图片用PPT展示了关键技术概念"
    }
]
```

## 注意事项

1. 需要先获取火山引擎ARK API Key
2. 使用Files API上传视频，支持最大512MB的视频文件
3. 支持的视频格式包括MP4、AVI、MOV等常见格式
4. 分析结果会自动保存为JSON格式文件
5. 抽帧频率（fps）参数控制分析精度，默认0.3帧/秒（每3秒抽1帧）
6. 上传的视频文件默认存储7天，可以在多次请求中重复使用