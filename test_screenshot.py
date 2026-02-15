import json
from video_analyzer_agent import VideoAnalyzerAgent

def test_screenshot():
    # 读取步骤分析结果
    with open('steps.json', 'r', encoding='utf-8') as f:
        steps = json.load(f)

    # 创建Agent实例
    agent = VideoAnalyzerAgent()

    # 测试截图功能
    video_path = '测试1.mp4'

    # 只测试前3个步骤
    test_steps = steps[:3]

    print(f"正在测试截图功能，将为前{len(test_steps)}个步骤生成截图...")
    screenshot_paths = agent.generate_screenshots_from_steps(video_path, test_steps)

    print(f"测试完成，生成了{len(screenshot_paths)}张截图")
    for path in screenshot_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    test_screenshot()
