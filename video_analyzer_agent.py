import os
import json
import asyncio
import logging
import subprocess
import re
import base64
import shutil
from pathlib import Path
from volcenginesdkarkruntime import AsyncArk
from typing import List, Dict, Optional
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
            self.api_key = os.getenv("ARK_API_KEY")
            if not self.api_key:
                raise ValueError("ARK_API_KEY 未设置，请在.env文件中设置或通过参数传入")

        self.client = AsyncArk(
            base_url='https://ark.cn-beijing.volces.com/api/v3',
            api_key=self.api_key
        )
        self.model = "doubao-seed-2-0-pro-260215"

        # Whisper 模型配置
        self.whisper_model = os.getenv("WHISPER_MODEL", "base")

    def _extract_response_text(self, response) -> str:
        """从模型响应中提取文本内容"""
        for item in response.output:
            if hasattr(item, 'role') and item.role == 'assistant' and hasattr(item, 'content'):
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        return content_item.text
        return ""

    # ========== Whisper 字幕生成 ==========

    def generate_subtitles(self, video_path: str, output_dir: str = ".") -> str:
        """
        调用本地 whisper 命令行从视频生成 SRT 字幕文件
        :param video_path: 视频文件路径
        :param output_dir: 字幕输出目录
        :return: 生成的 SRT 文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "whisper", str(video_path),
            "--model", self.whisper_model,
            "--language", "zh",
            "--output_format", "srt",
            "--output_dir", str(output_dir),
        ]
        print(f"正在使用 Whisper ({self.whisper_model}) 生成字幕...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Whisper 字幕生成失败: {result.stderr}")

        # whisper 输出文件名为 视频文件名.srt
        srt_filename = Path(video_path).stem + ".srt"
        srt_path = output_dir / srt_filename

        if not srt_path.exists():
            raise FileNotFoundError(f"Whisper 未生成字幕文件: {srt_path}")

        print(f"字幕已生成: {srt_path}")
        return str(srt_path)

    # ========== 字幕分析：识别操作步骤（默认模式，纯文本，便宜） ==========

    async def analyze_subtitles(self, srt_path: str) -> List[Dict]:
        """
        通过分析字幕文本识别操作步骤（Chat Completions API，纯文本调用）
        :param srt_path: SRT字幕文件路径
        :return: 操作步骤列表
        """
        subtitles = self.parse_srt(srt_path)
        subtitle_text = "\n".join(
            [f"[{sub['start_time']} --> {sub['end_time']}] {sub['text']}" for sub in subtitles]
        )

        system_prompt = """你是一个专业的操作视频分析助手。我会给你一段操作视频的字幕内容（带时间戳），请根据字幕的语义分析出视频中展示的所有操作步骤。

对于每个步骤，请提供：
1. step: 步骤编号（从1开始）
2. time: 该步骤最关键的时间点，格式为 "MM:SS"（选择该步骤中最能代表操作内容的一句话对应的时间）
3. title: 步骤标题（简洁，5-15个字）
4. description: 详细的操作说明（描述具体如何操作，30-100个字）
5. confidence: 你对这个步骤识别的自信度（0.0-1.0），评判标准：
   - 1.0: 字幕明确描述了具体操作，非常确定
   - 0.7-0.9: 字幕大致能推断操作，但细节不够清晰
   - 0.4-0.6: 字幕模糊，需要看画面才能确认具体操作
   - 0.0-0.3: 几乎无法从字幕判断，纯靠猜测

请按照以下JSON格式输出：
[
    {
        "step": 1,
        "time": "00:15",
        "title": "打开设置页面",
        "description": "点击屏幕左上角的菜单图标，在弹出的侧边栏中选择「设置」选项，进入系统设置页面。",
        "confidence": 0.9
    }
]

注意：
- 步骤数量没有限制，有多少步就输出多少步，不要人为合并或凑数，完全根据字幕内容如实拆分
- 时间格式必须为 "MM:SS"
- 根据字幕语义判断步骤的起止范围，time 取该步骤中最关键操作的时间点
- description 要具体、可操作，结合字幕内容让读者能照着做
- confidence 要诚实评估，不确定的地方就给低分
- 只输出JSON，不要添加其他文字"""

        user_prompt = f"""以下是操作视频的字幕内容：

{subtitle_text}

请分析字幕，识别出所有操作步骤。"""

        print("正在通过字幕分析识别操作步骤...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"  请求频率过快，等待 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        result = response.choices[0].message.content

        # 解析JSON
        try:
            steps = json.loads(result)
        except json.JSONDecodeError:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                steps = json.loads(json_match.group())
            else:
                raise ValueError(f"无法解析模型返回的JSON格式: {result}")

        return steps

    # ========== 视频分析：识别操作步骤（可选增强模式，上传视频，较贵） ==========

    async def analyze_video(self, video_path: str, fps: float = 1.0, file_id: str = None) -> List[Dict]:
        """
        分析视频，识别操作步骤
        :param video_path: 视频文件路径
        :param fps: 抽帧频率，默认1帧/秒
        :param file_id: 已上传的文件ID（可选，跳过上传）
        :return: 操作步骤列表
        """
        system_prompt = """你是一个专业的操作视频分析助手。请仔细观看上传的视频，识别出视频中展示的所有操作步骤。

对于每个步骤，请提供：
1. step: 步骤编号（从1开始）
2. time: 该步骤画面最具代表性的时间点（用于截图）
3. title: 步骤标题（简洁，5-15个字）
4. description: 详细的操作说明（描述具体如何操作，30-100个字）

请按照以下JSON格式输出：
[
    {
        "step": 1,
        "time": "00:15",
        "title": "打开设置页面",
        "description": "点击屏幕左上角的菜单图标，在弹出的侧边栏中选择「设置」选项，进入系统设置页面。"
    },
    {
        "step": 2,
        "time": "00:42",
        "title": "修改用户名",
        "description": "在设置页面中找到「个人信息」区域，点击用户名右侧的编辑按钮，输入新的用户名后点击保存。"
    }
]

注意：
- 时间格式必须为 "MM:SS"
- 选择每个步骤中最能展示操作内容的画面时间点
- description 要具体、可操作，让读者能照着做
- 只输出JSON，不要添加其他文字"""

        try:
            # 上传视频（如果没有提供 file_id）
            if not file_id:
                print(f"正在上传视频: {video_path}")
                file = await self.client.files.create(
                    file=open(video_path, "rb"),
                    purpose="user_data",
                    preprocess_configs={
                        "video": {"fps": fps}
                    }
                )
                file_id = file.id
                print(f"视频上传成功，File ID: {file_id}")

                await self.client.files.wait_for_processing(file_id)
                print(f"文件处理完成: {file_id}")
            else:
                print(f"使用已上传的文件: {file_id}")

            # 调用模型分析视频（带重试）
            print("正在分析视频，识别操作步骤...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "input_video", "file_id": file_id},
                                {"type": "input_text", "text": "请分析这个操作视频，识别出所有操作步骤"}
                            ]},
                        ]
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"  请求频率过快，等待 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            result = self._extract_response_text(response)

            # 解析JSON
            try:
                steps = json.loads(result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    steps = json.loads(json_match.group())
                else:
                    raise ValueError(f"无法解析模型返回的JSON格式: {result}")

            return steps

        except Exception as e:
            print(f"分析视频时出错: {e}")
            return []

    # ========== 截图生成 ==========

    def generate_screenshot(self, video_path: Path, output_dir: Path, timestamp: int, step_num: int = None) -> Path:
        """
        调用 ffmpeg 截图
        :param step_num: 步骤编号，如果提供则使用 step_XX 命名
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if step_num is not None:
            filename = f"step_{step_num:02d}.jpg"
        else:
            mm = timestamp // 60
            ss = timestamp % 60
            filename = f"screenshot_{mm:02d}_{ss:02d}.jpg"

        output_path = output_dir / filename

        cmd = [
            "ffmpeg", "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2",
            str(output_path), "-y",
        ]
        logging.info("生成截图：step=%s, time=%ss, file=%s", step_num, timestamp, output_path)
        subprocess.run(cmd, check=False, capture_output=True)
        return output_path

    def generate_screenshots_from_steps(self, video_path: str, steps: List[Dict], output_dir: str = "images") -> List[Path]:
        """
        根据步骤列表批量生成截图
        :return: 截图文件路径列表
        """
        video_path = Path(video_path)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        screenshot_paths = []
        for step in steps:
            time_str = step['time']
            parts = time_str.split(':')
            mm, ss = int(parts[0]), int(parts[1])
            timestamp = mm * 60 + ss

            step_num = step.get('step', len(screenshot_paths) + 1)
            screenshot_path = self.generate_screenshot(
                video_path, output_dir_path, timestamp, step_num=step_num
            )
            screenshot_paths.append(screenshot_path)
            print(f"已生成截图: {screenshot_path}")

        return screenshot_paths

    # ========== AI 看图增强（低自信度步骤） ==========

    async def enhance_steps_with_vision(
        self,
        steps: List[Dict],
        image_dir: str,
        srt_path: str = None,
        max_calls: int = 4
    ) -> List[Dict]:
        """
        对低自信度的步骤，调用 AI 看截图来增强描述
        :param steps: 步骤列表（含 confidence）
        :param image_dir: 截图目录
        :param srt_path: SRT字幕文件（可选，提供对应时间段的字幕给 AI 参考）
        :param max_calls: 最多调用 AI 看图的次数
        :return: 增强后的步骤列表
        """
        # 按 confidence 排序，取最低的 max_calls 个
        steps_with_idx = [(i, step) for i, step in enumerate(steps)]
        steps_with_idx.sort(key=lambda x: x[1].get('confidence', 1.0))
        to_enhance = steps_with_idx[:max_calls]

        # 解析字幕备用
        subtitles = []
        if srt_path and os.path.exists(srt_path):
            subtitles = self.parse_srt(srt_path)

        print(f"将对 {len(to_enhance)} 个低自信度步骤进行 AI 看图增强（最多 {max_calls} 次）")

        for idx, step in to_enhance:
            step_num = step.get('step', idx + 1)
            confidence = step.get('confidence', 0)
            img_path = Path(image_dir) / f"step_{step_num:02d}.jpg"

            if not img_path.exists():
                print(f"  步骤{step_num}: 截图不存在，跳过")
                continue

            # 读取图片转 base64
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"

            # 提取该步骤时间段内的字幕
            step_subtitle = ""
            if subtitles:
                time_str = step['time']
                parts = time_str.split(':')
                step_seconds = int(parts[0]) * 60 + int(parts[1])
                nearby = [s for s in subtitles if abs(s['start_seconds'] - step_seconds) < 30]
                if nearby:
                    step_subtitle = "\n".join([f"[{s['start_time']}] {s['text']}" for s in nearby])

            user_content = [
                {"type": "text", "text": f"这是操作视频第{step_num}步的截图。\n\n当前字幕分析结果：\n- 标题：{step['title']}\n- 描述：{step['description']}\n- 自信度：{confidence}"},
                {"type": "image_url", "image_url": {"url": img_data_url}},
            ]
            if step_subtitle:
                user_content.append({"type": "text", "text": f"\n该时间段附近的字幕：\n{step_subtitle}"})
            user_content.append({"type": "text", "text": "\n请根据截图画面，修正或补充这个步骤的标题和描述。只输出JSON，格式：{\"title\": \"...\", \"description\": \"...\"}"})

            print(f"  步骤{step_num} (confidence={confidence:.1f}): AI 看图分析中...")

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个专业的操作视频分析助手。请根据截图画面内容，修正或补充操作步骤的标题和描述。描述要具体、准确、可操作。只输出JSON。"},
                            {"role": "user", "content": user_content},
                        ],
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"    请求频率过快，等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"    AI 看图失败: {e}，保留原始结果")
                        break

            try:
                result = response.choices[0].message.content
                enhanced = json.loads(result)
                if not isinstance(enhanced, dict):
                    json_match = re.search(r'\{.*\}', result, re.DOTALL)
                    if json_match:
                        enhanced = json.loads(json_match.group())
                    else:
                        raise ValueError("无法解析")

                old_title = steps[idx]['title']
                steps[idx]['title'] = enhanced.get('title', steps[idx]['title'])
                steps[idx]['description'] = enhanced.get('description', steps[idx]['description'])
                steps[idx]['enhanced'] = True
                print(f"    ✓ 已增强: 「{old_title}」→「{steps[idx]['title']}」")
            except Exception as e:
                print(f"    解析增强结果失败: {e}，保留原始结果")

        return steps

    # ========== 步骤操作文档生成 ==========

    async def generate_step_document(
        self,
        steps: List[Dict],
        output_path: str = "operation_guide.md",
        srt_path: str = None,
        image_dir: str = "images",
        web_search: bool = False
    ) -> str:
        """
        生成步骤操作文档（Markdown格式）
        :param steps: AI识别出的步骤列表
        :param output_path: 输出文件路径
        :param srt_path: SRT字幕文件路径（可选，用于补充文字内容）
        :param image_dir: 截图目录
        :param web_search: 是否启用联网搜索增强
        """
        # 准备字幕信息（如果有）
        subtitle_text = ""
        if srt_path and os.path.exists(srt_path):
            subtitles = self.parse_srt(srt_path)
            subtitle_text = "\n".join(
                [f"[{sub['start_time']}] {sub['text']}" for sub in subtitles]
            )

        # 准备步骤和截图的对应关系
        steps_info = json.dumps(steps, ensure_ascii=False, indent=2)

        # 构建截图列表
        screenshot_list = []
        for step in steps:
            step_num = step.get('step', 0)
            filename = f"step_{step_num:02d}.jpg"
            screenshot_list.append(f"步骤{step_num}: ![步骤{step_num}截图]({image_dir}/{filename})")

        if web_search:
            system_prompt = """你是一个专业的技术文档编写专家。请根据提供的操作步骤信息、截图和字幕内容，生成一份清晰、专业的步骤操作文档。

你拥有联网搜索能力，请主动搜索以下内容来丰富文档：
- 视频中涉及的软件/平台/工具的官方介绍和功能说明
- 相关的最佳实践、使用技巧或注意事项
- 专业术语的准确解释

要求：
1. 使用 Markdown 格式
2. 每个步骤包含：标题（## 步骤 X：标题）、截图、详细操作说明
3. 操作说明要具体、准确，让读者能照着操作
4. 结合联网搜索到的信息，补充更丰富的上下文（如软件介绍、功能说明、注意事项等）
5. 如果有字幕内容，结合字幕让描述更加准确和详细
6. 在文档开头添加一个简短的概述（可结合搜索到的产品介绍）
7. 保持语言简洁专业
8. 在文档末尾添加「## 参考资料」章节，列出所有搜索引用的信息来源（标题+链接）
9. 直接返回 Markdown 内容，不要添加其他说明"""
        else:
            system_prompt = """你是一个专业的技术文档编写专家。请根据提供的操作步骤信息、截图和字幕内容，生成一份清晰、专业的步骤操作文档。

要求：
1. 使用 Markdown 格式
2. 每个步骤包含：标题（## 步骤 X：标题）、截图、详细操作说明
3. 操作说明要具体、准确，让读者能照着操作
4. 如果有字幕内容，结合字幕让描述更加准确和详细
5. 在文档开头添加一个简短的概述
6. 保持语言简洁专业
7. 直接返回 Markdown 内容，不要添加其他说明"""

        user_prompt = f"""操作步骤信息：
{steps_info}

截图对应关系：
{chr(10).join(screenshot_list)}
"""
        if subtitle_text:
            user_prompt += f"""
视频字幕内容（供参考）：
{subtitle_text}
"""
        if web_search:
            user_prompt += "\n请先联网搜索相关信息，然后生成完整的步骤操作文档。"
        else:
            user_prompt += "\n请生成完整的步骤操作文档。"

        if web_search:
            print("正在调用 AI 生成步骤操作文档（联网搜索增强）...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        tools=[{"type": "web_search"}],
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"  请求频率过快，等待 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            markdown_content = self._extract_response_text(response)
        else:
            print("正在调用 AI 生成步骤操作文档...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"  请求频率过快，等待 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            markdown_content = response.choices[0].message.content

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"步骤操作文档已保存到: {output_path}")
        return output_path

    # ========== PDF 生成 ==========

    def generate_pdf(self, md_path: str, pdf_path: str = None) -> str:
        """
        将 Markdown 文档转换为 PDF（图片嵌入）
        :param md_path: Markdown 文件路径
        :param pdf_path: PDF 输出路径（默认同名 .pdf）
        :return: PDF 文件路径
        """
        import markdown
        from weasyprint import HTML

        if not pdf_path:
            pdf_path = str(Path(md_path).with_suffix('.pdf'))

        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

        html_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.8; color: #333; }}
h1 {{ color: #1a1a1a; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; margin-top: 30px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
p {{ margin: 8px 0; }}
</style></head><body>{html_body}</body></html>"""

        # base_url 设为 md 文件所在目录，这样相对路径的图片能正确加载
        base_url = str(Path(md_path).parent.resolve())
        HTML(string=html_full, base_url=base_url).write_pdf(pdf_path)

        print(f"PDF 文档已生成: {pdf_path}")
        return pdf_path

    # ========== 工具方法 ==========

    def save_results(self, results: List[Dict], output_path: str):
        """将分析结果保存到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"分析结果已保存到: {output_path}")

    def parse_srt(self, srt_path):
        """解析srt文件，返回字幕列表"""
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        subtitle_blocks = re.split(r'\n\n+', content.strip())
        subtitles = []

        for block in subtitle_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_range = lines[1]
                    text = ' '.join(lines[2:])
                    start_time, end_time = time_range.split(' --> ')
                    start_seconds = self.time_to_seconds(start_time)
                    subtitles.append({
                        'index': index,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_seconds,
                        'text': text
                    })
                except Exception:
                    continue

        return subtitles

    def time_to_seconds(self, time_str):
        """将SRT时间字符串转换为秒数"""
        h, m, s = time_str.split(':')
        s, ms = s.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def find_first_file(ext: str) -> Optional[str]:
    """在当前目录查找第一个指定扩展名的文件"""
    import glob
    files = glob.glob(f'*.{ext}')
    return files[0] if files else None


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='操作视频 → 步骤操作文档生成器'
    )
    parser.add_argument('--api_key', help='火山引擎ARK API Key（可选，已在.env中设置则不需要）')
    parser.add_argument('--video_path', help='视频文件路径（默认自动查找当前目录MP4文件）')
    parser.add_argument('--srt_path', help='SRT字幕文件路径（可选，不提供则自动用Whisper生成）')
    parser.add_argument('--fps', type=float, default=1, help='抽帧频率，默认1帧/秒')
    parser.add_argument('--output_dir', help='输出目录（默认以视频文件名命名）')
    parser.add_argument('--whisper_model', default='base', help='Whisper模型，可选 tiny/base/small/medium/large')
    parser.add_argument('--use_video', action='store_true', help='启用视频上传分析模式（上传视频给AI看画面，较贵）')
    parser.add_argument('--file_id', help='已上传的视频文件ID（仅 --use_video 模式下有效）')
    parser.add_argument('--max_vision', type=int, default=4, help='最多对几个低自信度步骤调用 AI 看图增强（默认4）')
    parser.add_argument('--web_search', action='store_true', help='启用联网搜索增强文档内容（需要 API 支持）')

    args = parser.parse_args()

    # 确定视频文件路径
    video_path = args.video_path or find_first_file('mp4')
    if not video_path:
        print("错误：未找到MP4文件，请通过 --video_path 指定")
        return

    print(f"视频文件: {video_path}")

    # 创建输出目录
    output_dir = args.output_dir or Path(video_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 创建 Agent
    agent = VideoAnalyzerAgent(args.api_key)
    agent.whisper_model = args.whisper_model

    if args.use_video:
        # ===== 视频上传分析模式（较贵，可选） =====
        print("\n模式：视频上传分析（AI 看画面识别步骤）")

        # 步骤1（可选）：Whisper 生成字幕
        srt_path = args.srt_path or find_first_file('srt')
        if not srt_path:
            print("\n=== 步骤1：Whisper 生成字幕 ===")
            try:
                srt_path = agent.generate_subtitles(video_path)
            except Exception as e:
                print(f"Whisper 字幕生成失败: {e}，将继续不使用字幕")
                srt_path = None

        if srt_path:
            print(f"字幕文件: {srt_path}")

        # 步骤2：上传视频给 AI 分析
        print("\n=== 步骤2：上传视频，AI 识别操作步骤 ===")
        steps = await agent.analyze_video(video_path, args.fps, file_id=args.file_id)
    else:
        # ===== 字幕驱动模式（默认，便宜） =====
        print("\n模式：字幕驱动分析（AI 分析字幕文本识别步骤）")

        # 步骤1：确保有字幕
        srt_path = args.srt_path or find_first_file('srt')
        if not srt_path:
            print("\n=== 步骤1：Whisper 生成字幕 ===")
            srt_path = agent.generate_subtitles(video_path)

        print(f"字幕文件: {srt_path}")

        # 步骤2：AI 分析字幕文本，识别步骤
        print("\n=== 步骤2：分析字幕，识别操作步骤 ===")
        steps = await agent.analyze_subtitles(srt_path)

    # 将原视频和字幕复制到输出文件夹
    video_dest = output_dir / Path(video_path).name
    if not video_dest.exists():
        shutil.copy2(video_path, video_dest)
        print(f"已复制视频到: {video_dest}")
    if srt_path:
        srt_dest = output_dir / Path(srt_path).name
        if not srt_dest.exists():
            shutil.copy2(srt_path, srt_dest)
            print(f"已复制字幕到: {srt_dest}")

    if not steps:
        print("未能识别出操作步骤")
        return

    print(f"\n识别到 {len(steps)} 个操作步骤:")
    for step in steps:
        conf = step.get('confidence', '-')
        conf_str = f" (confidence={conf})" if conf != '-' else ""
        print(f"  步骤{step['step']}. [{step['time']}] {step['title']}{conf_str}")
        print(f"         {step['description']}")

    # 步骤3：生成截图
    print("\n=== 步骤3：生成截图 ===")
    agent.generate_screenshots_from_steps(video_path, steps, output_dir=str(image_dir))

    # 步骤3.5：AI 看图增强低自信度步骤
    if not args.use_video and args.max_vision > 0:
        print("\n=== 步骤3.5：AI 看图增强（低自信度步骤） ===")
        steps = await agent.enhance_steps_with_vision(
            steps, str(image_dir), srt_path=srt_path, max_calls=args.max_vision
        )

    # 保存步骤分析结果
    agent.save_results(steps, str(output_dir / "steps.json"))

    # 步骤4：生成步骤操作文档
    print("\n=== 步骤4：生成步骤操作文档 ===")
    output_md = output_dir / "operation_guide.md"
    await agent.generate_step_document(
        steps=steps,
        output_path=str(output_md),
        srt_path=srt_path,
        image_dir="images",
        web_search=args.web_search
    )

    # 步骤5：生成 PDF
    print("\n=== 步骤5：生成 PDF 文档 ===")
    output_pdf = output_dir / "operation_guide.pdf"
    agent.generate_pdf(str(output_md), str(output_pdf))

    print("\n=== 流程完成 ===")
    print(f"  输出目录: {output_dir}/")
    print(f"  原始视频: {video_dest}")
    if srt_path:
        print(f"  字幕文件: {output_dir / Path(srt_path).name}")
    print(f"  步骤数据: {output_dir}/steps.json")
    print(f"  截图目录: {image_dir}/")
    print(f"  操作文档: {output_md}")
    print(f"  PDF文档:  {output_pdf}")


if __name__ == "__main__":
    asyncio.run(main())
