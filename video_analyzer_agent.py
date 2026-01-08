import os
import json
import asyncio
import logging
import subprocess
import re
from pathlib import Path
from volcenginesdkarkruntime import AsyncArk
from typing import List, Dict
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoAnalyzerAgent:
    def __init__(self, api_key: str = None):
        """
        初始化视频分析AI Agent
        :param api_key: 火山引擎ARK API Key，如果为None则从.env文件读取
        """
        # 加载.env文件
        load_dotenv()
        
        if api_key:
            self.api_key = api_key
        else:
            # 从环境变量读取API Key
            self.api_key = os.getenv("ARK_API_KEY")
            if not self.api_key:
                raise ValueError("ARK_API_KEY 未设置，请在.env文件中设置或通过参数传入")
            
        self.client = AsyncArk(
            base_url='https://ark.cn-beijing.volces.com/api/v3',
            api_key=self.api_key
        )
        self.model = "doubao-seed-1-8-251228"
    
    async def analyze_video(self, video_path: str, fps: float = 0.3) -> List[Dict]:
        """
        使用doubao-seed-1.8模型分析视频，提取关键图片位置
        :param video_path: 视频文件路径
        :param fps: 抽帧频率，默认0.3帧/秒（每3秒抽1帧）
        :return: 包含关键图片时间和原因的列表
        """
        # 构建系统提示词
        system_prompt = """
        你是一个专业的视频分析AI助手，请分析上传的视频，找出所有包含以下内容的关键位置：
        1. 代码讲解的画面
        2. 关键技术讲解的PPT页面
        3. 能有效帮助理解视频内容的重要图片
        4. 其他具有高信息量的关键画面
        
        请按照以下JSON格式输出结果：
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
        
        注意：
        - 时间格式必须为"MM:SS"
        - reason要简洁明了，准确描述图片内容
        - 只输出JSON格式，不要添加其他解释性文字
        """
        
        try:
            # 1. 上传视频文件
            # file_id='file-20251212220246-fvrh2'
            file_id=None
            if not file_id:
                print(f"正在上传视频: {video_path}")
                file = await self.client.files.create(
                    file=open(video_path, "rb"),
                    purpose="user_data",
                    preprocess_configs={
                        "video": {
                            "fps": fps,
                        }
                    }
                )
                print(f"视频上传成功，File ID: {file.id}")
                
                # 等待文件处理完成
                await self.client.files.wait_for_processing(file.id)
                print(f"文件处理完成: {file.id}")
                file_id = file.id
            
            # 2. 调用Responses API分析视频
            print("正在分析视频...")
            response   = await self.client.responses.create(
                model=self.model,
                input =[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {
                            "type": "input_video",
                            "file_id":file_id
                        },
                        {
                            "type": "input_text",
                            "text": "请分析这个视频，找出所有关键图片位置"
                        }
                    ]},
                ]
            )
            
            # 解析模型返回结果
            # 找到assistant的message响应
            for item in response.output:
                if hasattr(item, 'role') and item.role == 'assistant' and hasattr(item, 'content'):
                    # 遍历content找到text内容
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            result = content_item.text
                            break
                    break
            
            # 尝试解析JSON
            try:
                key_frames = json.loads(result)
                return key_frames
            except json.JSONDecodeError:
                # 如果模型返回的不是纯JSON，尝试提取JSON部分
                import re
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("无法解析模型返回的JSON格式")
                    
        except Exception as e:
            print(f"分析视频时出错: {e}")
            return []
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        将分析结果保存到JSON文件
        :param results: 分析结果
        :param output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"分析结果已保存到: {output_path}")
    
    def generate_screenshot(self, video_path: Path, output_dir: Path, timestamp: int) -> Path:
        """
        调用 ffmpeg 截图，返回图片路径。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        mm = timestamp // 60
        ss = timestamp % 60
        filename = f"screenshot_{mm:02d}_{ss:02d}.jpg"
        output_path = output_dir / filename

        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
            "-y",
        ]
        logging.info("生成截图：time=%s, file=%s", timestamp, output_path)
        subprocess.run(cmd, check=False, capture_output=True)
        return output_path
    
    def generate_screenshots_from_results(self, video_path: str, results: List[Dict], output_dir: str = "images") -> List[Path]:
        """
        根据分析结果批量生成截图
        :param video_path: 视频文件路径
        :param results: 分析结果
        :param output_dir: 截图输出目录
        :return: 生成的截图文件路径列表
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        screenshot_paths = []
        
        for frame in results:
            time_str = frame['time']
            # 将时间格式从"MM:SS"转换为秒数
            mm, ss = map(int, time_str.split(':'))
            timestamp = mm * 60 + ss
            
            screenshot_path = self.generate_screenshot(video_path, output_dir, timestamp)
            screenshot_paths.append(screenshot_path)
            print(f"已生成截图: {screenshot_path}")
        
        return screenshot_paths
    
    def parse_srt(self, srt_path):
        """解析srt文件，返回字幕列表"""
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割字幕块
        subtitle_blocks = re.split(r'\n\n+', content.strip())
        subtitles = []
        
        for block in subtitle_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_range = lines[1]
                    text = ' '.join(lines[2:])
                    
                    # 解析时间
                    start_time, end_time = time_range.split(' --> ')
                    start_seconds = self.time_to_seconds(start_time)
                    
                    subtitles.append({
                        'index': index,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_seconds,
                        'text': text
                    })
                except:
                    continue
        
        return subtitles
    
    def time_to_seconds(self, time_str):
        """将时间字符串转换为秒数"""
        h, m, s = time_str.split(':')
        s, ms = s.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    
    async def convert_srt_to_markdown(self, srt_path, output_path="video_notes.md"):
        """将srt文件转换为带插图的markdown笔记"""
        # 解析srt文件
        subtitles = self.parse_srt(srt_path)
        print(f"解析到 {len(subtitles)} 条字幕")
        
        # 获取所有截图
        screenshot_dir = "images"
        screenshots = []
        if os.path.exists(screenshot_dir):
            screenshots = sorted([f for f in os.listdir(screenshot_dir) if f.endswith('.jpg')])
        
        # 准备截图信息
        screenshot_info = []
        for screenshot in screenshots:
            match = re.search(r'screenshot_(\d{2})_(\d{2})\.jpg', screenshot)
            if match:
                mm = int(match.group(1))
                ss = int(match.group(2))
                screenshot_time = mm * 60 + ss
                screenshot_info.append({
                    'time': screenshot_time,
                    'filename': screenshot,
                    'markdown': f"![截图]({screenshot_dir}/{screenshot})"
                })
        
        # 准备所有字幕文本
        all_subtitles = "\n".join([f"[{subtitle['start_time']}] {subtitle['text']}" for subtitle in subtitles])
        
        # 构建系统提示词
        system_prompt = """
        你是一个专业的视频笔记整理专家，请将以下字幕内容整理成一篇结构清晰、可读性强的Markdown笔记。
        
        要求：
        1. 为字幕添加合适的标点符号，保持原意不变
        2. 将字幕内容分段，形成自然的段落结构
        3. 根据提供的截图信息，在合适的位置插入对应的图片
        4. 图片应该插入到与其时间最接近的字幕内容附近
        5. 保持Markdown格式简洁美观，使用合适的标题和段落
        6. 不要添加任何与视频内容无关的解释性文字
        
        截图信息格式：
        - 时间（秒）: 图片Markdown代码
        
        请直接返回最终的Markdown笔记内容，不要添加任何其他说明。
        """
        
        # 构建用户提示词
        user_prompt = f"""
        字幕内容：
        {all_subtitles}
        
        截图信息：
        {chr(10).join([f"- {s['time']}秒: {s['markdown']}" for s in screenshot_info])}
        
        请根据以上内容生成完整的Markdown笔记。
        """

        print("系统提示词")
        print(system_prompt)
        print("用户提示词")
        print(user_prompt)

        print("正在调用AI生成完整的Markdown笔记...")
        response = await self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # 解析返回结果
        markdown_content = ""
        for item in response.output:
            if hasattr(item, 'role') and item.role == 'assistant' and hasattr(item, 'content'):
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        markdown_content = content_item.text
                        break
                break
        
        # 保存markdown文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Markdown笔记已保存到: {output_path}")
        return output_path

def find_first_mp4() -> str:
    """
    在当前目录查找第一个MP4文件
    :return: 找到的MP4文件路径，未找到则返回None
    """
    import glob
    mp4_files = glob.glob('*.mp4')
    if mp4_files:
        return mp4_files[0]
    return None

async def main():
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='视频分析AI Agent - 完整流程：分析视频→截图→生成markdown笔记')
    parser.add_argument('--api_key', help='火山引擎ARK API Key（可选，如果已在.env文件中设置）')
    parser.add_argument('--video_path', help='视频文件路径（可选，默认自动查找当前目录第一个MP4文件）')
    parser.add_argument('--srt_path', help='srt字幕文件路径（可选，默认查找当前目录第一个srt文件）')
    parser.add_argument('--fps', type=float, default=1, help='抽帧频率，默认1帧/秒（每1秒抽1帧）')
    parser.add_argument('--output', default='results.json', help='输出结果文件路径')
    parser.add_argument('--markdown_output', default='video_notes.md', help='markdown笔记输出路径')
    
    args = parser.parse_args()
    
    # 确定视频文件路径
    video_path = args.video_path
    if not video_path:
        video_path = find_first_mp4()
        if not video_path:
            print("错误：未找到MP4文件，请指定视频文件路径")
            return
    
    # 确定srt文件路径
    srt_path = args.srt_path
    if not srt_path:
        import glob
        srt_files = glob.glob('*.srt')
        if srt_files:
            srt_path = srt_files[0]
            print(f"自动找到srt文件: {srt_path}")
        else:
            print("错误：未找到srt文件，请指定srt文件路径")
            return
    
    # 创建视频分析Agent
    agent = VideoAnalyzerAgent(args.api_key)
    
    # 步骤1：分析视频
    print("=== 步骤1：分析视频 ===")
    results = await agent.analyze_video(video_path, args.fps)
    
    # 输出结果
    if results:
        print("\n分析结果:")
        for i, frame in enumerate(results, 1):
            print(f"{i}. 时间: {frame['time']}, 原因: {frame['reason']}")
        
        # 保存结果
        agent.save_results(results, args.output)
        
        # 步骤2：生成截图
        print("\n=== 步骤2：生成截图 ===")
        agent.generate_screenshots_from_results(video_path, results)
        print(f"截图已保存到 images 目录")
        
        # 步骤3：将srt转换为带插图的markdown笔记
        print("\n=== 步骤3：生成markdown笔记 ===")
        await agent.convert_srt_to_markdown(srt_path, args.markdown_output)
        print("\n=== 流程完成 ===")
    else:
        print("未找到关键图片位置")

if __name__ == "__main__":
    asyncio.run(main())
