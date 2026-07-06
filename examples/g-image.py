"""
Nebula Gemini 图生图最小示例
对应节点：🌌 Nebula Gemini（API_nebula.py 的 NebulaGeminiNode）

关键点（对方 demo 确认的标准用法）：
- 参考图放扁平的 `image` 字段（单图）/ `images` 数组（多图）
- 不要放 `contents[].parts[]` 里
- prompt 同时放顶层

使用方法：
1. pip install requests
2. 设置环境变量 NEBULA_API_KEY（或在下方直接填写）
3. 准备一张输入图片，修改 INPUT_IMAGE 路径
4. python g-image.py
"""

import base64
import os
import requests

# ===== 配置 =====
API_KEY = os.environ.get("NEBULA_API_KEY", "your-api-key-here")
INPUT_IMAGE = "./image/dog.png"        # 输入图片路径
OUTPUT_IMAGE = "./output.jpg"          # 输出图片路径
MODEL = "gemini-3.1-flash-image-preview"
PROMPT = "给小狗带上圣诞帽子"

url = "https://llm.ai-nebula.com/v1/images/generations"

with open(INPUT_IMAGE, "rb") as f:
    images_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": MODEL,
    "prompt": PROMPT,
    "image": f"data:image/png;base64,{images_b64}",   # 单图用 image 字段
    # 多图融合时改用： "images": ["data:image/png;base64,...", ...]
    "response_format": "b64_json",
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

response = requests.post(url, json=payload, headers=headers, timeout=600).json()

base64_str = response["data"][0]["b64_json"]
with open(OUTPUT_IMAGE, "wb") as f:
    f.write(base64.b64decode(base64_str))

print(f"✅ 生成完成：{OUTPUT_IMAGE}")
