# ComfyUI AFOLIE API - Nebula 图片生成节点

🌌 支持多种 AI 图像生成模型的 ComfyUI 自定义节点插件。

## 功能特性

- ✅ 支持 **Gemini** 系列模型（gemini-2.5-flash-image、gemini-3-pro-image-preview）
- ✅ 支持 **豆包 Seedream** 系列模型（3.0、4.0、4.5）
- ✅ 支持 **GPT Image** 系列模型（gpt-image-1、gpt-image-1-mini）
- ✅ 支持 **通义千问** 系列模型（qwen-image-plus、qwen-image-edit-plus）
- ✅ 支持 **OpenAI 协议**（DALL-E、Flux 等）
- ✅ 文生图、图生图、多图融合
- ✅ 双协议支持（Gemini/OpenAI）
- ✅ 批量生成（1-8张图片）
- ✅ 最多支持 9 张参考图像
- ✅ 中文界面，易于使用
- ✅ 支持多种图片尺寸和质量设置

## 安装

### 方法一：手动安装

1. 将 `comfyui-AFOLIE-API` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录
2. 重启 ComfyUI

### 方法二：Git 克隆

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/comfyui-AFOLIE-API.git
```

## 配置

编辑 `config.ini` 文件，填入你的 API Key：

```ini
[nebula]
api_key = sk-your-api-key-here
api_base_url = https://llm.ai-nebula.com/v1
max_workers = 4
```

或者在节点中直接输入 API Key。

## 节点说明

### 🌌 Nebula 图像生成

通用图像生成节点，支持所有模型。

**输入参数：**
- `提示词`：生成图像的文本描述
- `模型`：选择图像生成模型
- `API密钥`：API Key（可选，留空使用配置文件）
- `图片尺寸`：输出图片尺寸
- `图片质量`：图片质量设置
- `生成数量`：生成图片数量
- `参考图像`：用于图生图的参考图像（可选）
- `负面提示词`：排除不想要的元素（可选）
- `随机种子`：控制生成结果的随机性

### 🌌 Nebula Gemini

专门针对 Gemini 模型优化的节点。

**特有参数：**
- `宽高比`：1:1、16:9、9:16 等
- `图片尺寸`：1K、2K、4K
- 支持最多 3 张参考图像进行多图融合

### 🌌 Nebula 豆包 Seedream

豆包 Seedream 系列模型专用节点。

**特有参数：**
- `水印`：是否添加水印
- `引导系数`：控制生成图像与提示词的匹配程度（3.0 模型）
- `优化模式`：standard、fast、creative、precise（4.x 模型）

### 🌌 Nebula GPT Image

GPT Image 系列模型专用节点。

**特有参数：**
- `图片质量`：low、medium、high
- `输入保真度`：控制参考图像的保真度
- 支持最多 10 张图片输入

### 🌌 Nebula 通义千问

通义千问系列模型专用节点。

**特有参数：**
- `负面提示词`：排除不想要的元素
- `提示词扩展`：自动扩展简短提示词
- `水印`：是否添加水印

### 🎨 AFOLIE 高级图像生成

功能最全面的高级图像生成节点，支持双协议和批量生成。

**必需参数：**
- `prompt`：文本提示词，支持多行输入
- `api_key`：API 密钥（可选，可用 config.ini 配置）
- `api_base_url`：API 服务地址（可选）
- `model_type`：模型名称
- `batch_size`：批次大小（1-8），一次生成图片数量
- `aspect_ratio`：宽高比（Auto/1:1/16:9/9:16/2:3/3:2/4:3/3:4/4:5/5:4/21:9）

**可选参数：**
- `seed`：随机种子（-1 为随机，固定值可复现）
- `top_p`：采样参数（0.0-1.0），控制多样性
- `imageSize`：分辨率（无/1K/2K/4K）
- `image_1~9`：最多支持 9 张参考图像（图生图）
- `超时秒数`：API 请求超时时间（0-1800秒）
- `绕过代理`：梯子不稳定时开启
- `api_protocol`：选择 Gemini 或 OpenAI 协议

**特性：**
- 🔄 双协议支持：Gemini 和 OpenAI 协议自动适配
- 📦 批量生成：可配置 1-8 张图片同时生成
- 🖼️ 多图参考：支持最多 9 张输入图像
- 🚀 并发处理：使用线程池提高生成效率
- 🎯 智能参数：自动检测宽高比、自动添加参数到提示词
- ⚙️ 错误处理：详细的错误信息、部分失败可继续
- 📊 进度显示：实时更新生成进度
- 🔧 中断支持：集成 ComfyUI 原生取消机制

## 支持的模型

### Gemini 系列
| 模型 | 说明 |
|------|------|
| gemini-2.5-flash-image | Nano Banana，支持文生图、图生图、多图融合 |
| gemini-3-pro-image-preview | Nano Banana Pro，更高质量输出 |

### 豆包 Seedream 系列
| 模型 | 说明 |
|------|------|
| doubao-seedream-3-0-t2i-250415 | 3.0 文生图，支持引导系数 |
| doubao-seedream-4-0-250828 | 4.0 版本，支持 2K/4K |
| doubao-seedream-4-5-251128 | 4.5 版本，支持创意/精确模式 |
| doubao-seededit-3-0-i2i-250628 | 图片编辑模型 |

### GPT Image 系列
| 模型 | 说明 |
|------|------|
| gpt-image-1 | 高质量图像生成 |
| gpt-image-1-mini | 快速生成，成本更低 |

### 通义千问系列
| 模型 | 说明 |
|------|------|
| qwen-image-plus | 文生图，擅长中文文本渲染 |
| qwen-image-edit-plus | 图片编辑 |

### 其他模型
| 模型 | 说明 |
|------|------|
| dalle-3 | OpenAI DALL-E 3 |
| dall-e-3 | OpenAI DALL-E 3 |
| flux-pro | Flux 专业版 |
| flux-dev | Flux 开发版 |
| sdxl-turbo | Stable Diffusion XL Turbo |

## 使用示例

### 文生图

1. 添加 `🌌 Nebula 图像生成` 节点
2. 输入提示词，如："一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影"
3. 选择模型和尺寸
4. 连接到 `Preview Image` 或 `Save Image` 节点
5. 运行工作流

### 图生图

1. 添加 `🌌 Nebula Gemini` 节点
2. 连接参考图像到 `参考图像1` 输入
3. 输入提示词描述想要的修改
4. 运行工作流

### 多图融合（Gemini）

1. 添加 `🌌 Nebula Gemini` 节点
2. 连接多张参考图像
3. 输入提示词，如："将第一张图的风格应用到第二张图的内容上"
4. 运行工作流

## 常见问题

### Q: API Key 从哪里获取？
A: 请访问 [Nebula API](https://ai-nebula.com) 注册并获取 API Key。

### Q: 生成失败怎么办？
A: 请检查：
1. API Key 是否正确
2. 网络连接是否正常
3. 模型是否支持当前参数设置

### Q: 如何提高生成质量？
A: 
1. 使用详细的提示词描述
2. 选择更高的图片质量设置
3. 尝试不同的模型

## 更新日志

### v1.1.0
- 新增 🎨 AFOLIE 高级图像生成节点
- 支持 Gemini 和 OpenAI 双协议
- 支持批量生成（1-8张图片）
- 支持最多 9 张参考图像输入
- 新增并发处理和线程池支持
- 新增智能参数处理和宽高比自动检测
- 新增详细的错误处理和进度显示

### v1.0.0
- 初始版本
- 支持 Gemini、豆包 Seedream、GPT Image、通义千问模型
- 支持文生图、图生图功能

## 许可证

MIT License

## 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Nebula API](https://ai-nebula.com)
