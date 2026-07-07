"""
Nebula 图片生成节点
支持 Gemini、豆包 Seedream、GPT Image、通义千问等多种模型
"""

from __future__ import annotations

import json
import torch
from typing import List, Dict, Optional, Tuple, Any
import time
import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

try:
    from server import PromptServer
except ImportError:
    class _DummyPromptServer:
        instance = None
    PromptServer = _DummyPromptServer()

import comfy.utils
import comfy.model_management

from nebula_logger import logger
from nebula_config_manager import ConfigManager
from nebula_image_codec import ImageCodec, ErrorCanvas
from nebula_api_client import NebulaApiClient


CONFIG_MANAGER = ConfigManager(MODULE_DIR)
API_CLIENT = NebulaApiClient(
    CONFIG_MANAGER,
    logger,
    interrupt_checker=comfy.model_management.throw_exception_if_processing_interrupted,
)


# 模型列表
GEMINI_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
]

DOUBAO_MODELS = [
    "doubao-seedream-3-0-t2i-250415",
    "doubao-seedream-4-0-250828",
    "doubao-seedream-4-5-251128",
    "doubao-seededit-3-0-i2i-250628",
]

GPT_MODELS = [
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
]

QWEN_MODELS = [
    "qwen-image-plus",
    "qwen-image-edit-plus",
]

ALL_MODELS = GEMINI_MODELS + DOUBAO_MODELS + GPT_MODELS + QWEN_MODELS


class NebulaImageGenerator:
    """
    ComfyUI节点: Nebula 图像生成
    支持多种 AI 图像生成模型
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "信息")
    FUNCTION = "generate_images"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/nebula图像节点"

    def __init__(self):
        self.config_manager = CONFIG_MANAGER
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影",
                    "tooltip": "生成图像的文本提示词"
                }),
                "模型": (ALL_MODELS, {
                    "default": "gemini-2.5-flash-image",
                    "tooltip": "选择图像生成模型"
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API Key，留空则使用 config.ini 中的配置"
                }),
            },
            "optional": {
                "图片尺寸": ("STRING", {
                    "default": "1024x1024",
                    "multiline": False,
                    "tooltip": "图片尺寸，如 1024x1024、16:9、2048x2048 等"
                }),
                "图片质量": (["auto", "low", "medium", "high", "hd", "1K", "2K", "3K"], {
                    "default": "high",
                    "tooltip": "图片质量设置"
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "生成图片数量"
                }),
                "参考图像": ("IMAGE", {
                    "tooltip": "参考图像，用于图生图"
                }),
                "负面提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "负面提示词，用于排除不想要的元素"
                }),
                "随机种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "随机种子，-1 为随机"
                }),
                "超时秒数": ("INT", {
                    "default": 420,
                    "min": 30,
                    "max": 1800,
                    "tooltip": "API 请求超时时间（秒）"
                }),
                "水印": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印（豆包/通义千问）"
                }),
                "提示词扩展": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用提示词扩展（通义千问）"
                }),
                "引导系数": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "引导系数（豆包 Seedream 3.0）"
                }),
                "优化模式": (["standard", "fast", "creative", "precise"], {
                    "default": "standard",
                    "tooltip": "提示词优化模式（豆包 Seedream 4.x）"
                }),
                "输入保真度": (["auto", "low", "medium", "high"], {
                    "default": "medium",
                    "tooltip": "输入图片保真度（GPT Image）"
                }),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def _get_model_category(self, model: str) -> str:
        """获取模型类别"""
        if model in GEMINI_MODELS:
            return "gemini"
        elif model in DOUBAO_MODELS:
            return "doubao"
        elif model in GPT_MODELS:
            return "gpt"
        elif model in QWEN_MODELS:
            return "qwen"
        return "unknown"

    def _build_extra_params(
        self,
        model: str,
        negative_prompt: str,
        seed: int,
        watermark: bool,
        prompt_extend: bool,
        guidance_scale: float,
        optimize_mode: str,
        input_fidelity: str,
    ) -> Dict[str, Any]:
        """根据模型类型构建额外参数"""
        extra_params = {}
        category = self._get_model_category(model)

        if category == "gemini":
            # Gemini 参数
            if seed >= 0:
                extra_params["seed"] = seed

        elif category == "doubao":
            # 豆包 Seedream 参数
            extra_params["watermark"] = watermark
            if seed >= 0:
                extra_params["seed"] = seed
            
            # 3.0 模型支持引导系数
            if "3-0" in model:
                extra_params["guidance_scale"] = guidance_scale
            
            # 4.x 模型支持优化模式
            if "4-0" in model or "4-5" in model:
                extra_params["optimize_prompt_options"] = {"mode": optimize_mode}

        elif category == "gpt":
            # GPT Image 参数
            extra_params["input_fidelity"] = input_fidelity

        elif category == "qwen":
            # 通义千问参数
            params = {
                "watermark": watermark,
                "prompt_extend": prompt_extend,
            }
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
            if seed >= 0:
                params["seed"] = seed
            extra_params["parameters"] = params

        return extra_params

    def generate_images(
        self,
        提示词: str,
        模型: str,
        API密钥: str = "",
        图片尺寸: str = "1024x1024",
        图片质量: str = "high",
        生成数量: int = 1,
        参考图像: Optional[torch.Tensor] = None,
        负面提示词: str = "",
        随机种子: int = -1,
        超时秒数: int = 420,
        水印: bool = False,
        提示词扩展: bool = True,
        引导系数: float = 2.5,
        优化模式: str = "standard",
        输入保真度: str = "medium",
    ):
        """生成图像"""
        start_time = time.time()

        # 解析 API Key
        raw_api_key = (API密钥 or "").strip()
        resolved_api_key = self.config_manager.sanitize_api_key(raw_api_key)
        if not resolved_api_key:
            resolved_api_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_api_key:
            error_msg = "请在 config.ini 中配置 API Key 或在节点中填写"
            logger.error(error_msg)
            error_tensor = self.error_canvas.build_error_tensor_from_text(
                "配置缺失",
                f"{error_msg}\n请在 config.ini 或节点输入中填写有效 API Key"
            )
            return (error_tensor, error_msg)

        # 获取 API Base URL
        api_base_url = self.config_manager.get_effective_api_base_url()

        # 输出配置信息
        masked_key = resolved_api_key[:8] + "..." + resolved_api_key[-4:] if len(resolved_api_key) > 12 else "***"
        logger.info(f"使用 API Base URL: {api_base_url}")
        logger.info(f"使用 API Key: {masked_key}")
        logger.info(f"使用模型: {模型}")

        # 准备输入图像
        input_images_b64 = []
        if 参考图像 is not None:
            input_images_b64 = self.image_codec.prepare_input_images([参考图像])

        # 构建额外参数
        extra_params = self._build_extra_params(
            模型, 负面提示词, 随机种子, 水印, 提示词扩展,
            引导系数, 优化模式, 输入保真度
        )

        # 显示任务信息
        logger.header("🌌 Nebula 图像生成任务")
        logger.info(f"模型: {模型}")
        logger.info(f"尺寸: {图片尺寸}")
        logger.info(f"质量: {图片质量}")
        logger.info(f"数量: {生成数量}")
        if 随机种子 >= 0:
            logger.info(f"种子: {随机种子}")
        logger.separator()

        try:
            self._ensure_not_interrupted()

            # 创建请求数据
            request_data = API_CLIENT.create_request_data(
                model=模型,
                prompt=提示词,
                size=图片尺寸,
                quality=图片质量,
                n=生成数量,
                response_format="b64_json",
                input_images_b64=input_images_b64 if input_images_b64 else None,
                **extra_params
            )

            self._ensure_not_interrupted()

            # 发送请求
            response_data = API_CLIENT.send_request(
                resolved_api_key,
                request_data,
                api_base_url,
                timeout=(15, 超时秒数),
                bypass_proxy=False,
                verify_ssl=True,
            )

            self._ensure_not_interrupted()

            # 提取图片
            base64_images, revised_prompt = API_CLIENT.extract_images(response_data)

            if not base64_images:
                error_msg = "API 未返回任何图片"
                logger.warning(error_msg)
                error_tensor = self.error_canvas.build_error_tensor_from_text(
                    "生成失败", error_msg
                )
                return (error_tensor, error_msg)

            # 解码图片
            self._ensure_not_interrupted()
            image_tensor = self.image_codec.base64_to_tensor_parallel(
                base64_images,
                log_prefix="Nebula",
                max_workers=self.config_manager.load_max_workers()
            )

            total_time = time.time() - start_time
            actual_count = len(base64_images)
            avg_time = total_time / actual_count if actual_count > 0 else 0

            # 构建返回信息
            info_text = f"✅ 成功生成 {actual_count} 张图像\n"
            info_text += f"模型: {模型}\n"
            info_text += f"尺寸: {图片尺寸}\n"
            info_text += f"总耗时: {total_time:.2f}s，平均 {avg_time:.2f}s/张"
            if revised_prompt:
                info_text += f"\n修订提示词: {revised_prompt}"

            # 显示完成统计
            logger.summary("任务完成", {
                "生成数量": f"{actual_count} 张",
                "总耗时": f"{total_time:.2f}s",
                "平均速度": f"{avg_time:.2f}s/张"
            })

            return (image_tensor, info_text)

        except comfy.model_management.InterruptProcessingException:
            logger.warning("任务已取消")
            raise
        except Exception as e:
            error_msg = str(e)[:500]
            logger.error(f"生成失败: {error_msg}")
            error_tensor = self.error_canvas.build_error_tensor_from_text(
                "生成失败", error_msg
            )
            return (error_tensor, error_msg)


class NebulaGeminiNode:
    """
    Gemini 图像生成节点
    专门针对 Gemini 模型优化的节点
    支持动态参考图像输入端口
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "信息")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/nebula图像节点"

    def __init__(self):
        self.config_manager = CONFIG_MANAGER
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影",
                }),
                "模型": (GEMINI_MODELS, {
                    "default": "gemini-3.1-flash-image-preview",
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "宽高比": (["Auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                    "default": "Auto",
                    "tooltip": "Auto 会自动匹配参考图像的相近宽高比"
                }),
                "图片尺寸": (["1K", "2K", "4K"], {
                    "default": "2K",
                    "tooltip": "输出图片分辨率：1K/2K/4K"
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "超时秒数": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1800,
                    "tooltip": "API 请求超时时间（秒），0 表示无限等待"
                }),
                "参考图像1": ("IMAGE", {"tooltip": "参考图像 1"}),
                "参考图像2": ("IMAGE", {"tooltip": "参考图像 2"}),
                "参考图像3": ("IMAGE", {"tooltip": "参考图像 3"}),
                "参考图像4": ("IMAGE", {"tooltip": "参考图像 4"}),
                "参考图像5": ("IMAGE", {"tooltip": "参考图像 5"}),
                "参考图像6": ("IMAGE", {"tooltip": "参考图像 6"}),
                "参考图像7": ("IMAGE", {"tooltip": "参考图像 7"}),
                "参考图像8": ("IMAGE", {"tooltip": "参考图像 8"}),
                "参考图像9": ("IMAGE", {"tooltip": "参考图像 9"}),
                "参考图像10": ("IMAGE", {"tooltip": "参考图像 10"}),
                "参考图像11": ("IMAGE", {"tooltip": "参考图像 11"}),
                "参考图像12": ("IMAGE", {"tooltip": "参考图像 12"}),
                "参考图像13": ("IMAGE", {"tooltip": "参考图像 13"}),
                "参考图像14": ("IMAGE", {"tooltip": "参考图像 14"}),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def generate(
        self,
        提示词: str,
        模型: str,
        API密钥: str = "",
        宽高比: str = "Auto",
        图片尺寸: str = "2K",
        生成数量: int = 1,
        超时秒数: int = 0,
        **kwargs
    ):
        """生成 Gemini 图像"""
        start_time = time.time()

        # 解析 API Key
        raw_api_key = (API密钥 or "").strip()
        resolved_api_key = self.config_manager.sanitize_api_key(raw_api_key)
        if not resolved_api_key:
            resolved_api_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_api_key:
            error_msg = "请配置 API Key"
            error_tensor = self.error_canvas.build_error_tensor_from_text("配置缺失", error_msg)
            return (error_tensor, error_msg)

        api_base_url = self.config_manager.get_effective_api_base_url()

        # 收集所有参考图像
        input_tensors = []
        for i in range(1, 15):
            img = kwargs.get(f"参考图像{i}")
            if img is not None:
                input_tensors.append(img)
        
        input_images_b64 = self.image_codec.prepare_input_images(input_tensors) if input_tensors else []
        
        # 自动检测宽高比
        effective_aspect_ratio = 宽高比
        if 宽高比 == "Auto" and input_tensors:
            # 从第一张参考图像获取宽高比
            first_img = input_tensors[0]
            if first_img is not None and len(first_img.shape) >= 3:
                h, w = first_img.shape[1], first_img.shape[2]
                ratio = w / h
                # 匹配最接近的宽高比
                aspect_ratios = {
                    "1:1": 1.0, "3:2": 1.5, "2:3": 0.667, "3:4": 0.75, "4:3": 1.333,
                    "4:5": 0.8, "5:4": 1.25, "9:16": 0.5625, "16:9": 1.778, "21:9": 2.333
                }
                closest = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
                effective_aspect_ratio = closest[0]
                logger.info(f"自动检测宽高比: {effective_aspect_ratio} (原始比例: {ratio:.2f})")

        logger.header("🌌 Gemini 图像生成")
        logger.info(f"模型: {模型}")
        logger.info(f"宽高比: {effective_aspect_ratio}")
        logger.info(f"图片尺寸: {图片尺寸}")
        if input_tensors:
            logger.info(f"参考图像: {len(input_tensors)} 张")

        # 处理超时：0 表示无限等待
        timeout_value = None if 超时秒数 == 0 else 超时秒数

        try:
            self._ensure_not_interrupted()

            if input_images_b64:
                # 图生图：参考图放扁平的 image(单图)/ images(数组) 字段，
                # 不走 contents[].parts[] 格式（对方 demo 确认的标准用法）
                request_data = {
                    "model": 模型,
                    "prompt": 提示词,
                    "n": 生成数量,
                    "response_format": "b64_json",
                    "image_size": 图片尺寸,
                }
                if effective_aspect_ratio != "Auto":
                    request_data["size"] = effective_aspect_ratio
                if len(input_images_b64) == 1:
                    request_data["image"] = f"data:image/png;base64,{input_images_b64[0]}"
                else:
                    request_data["images"] = [
                        f"data:image/png;base64,{b64}" for b64 in input_images_b64
                    ]
            else:
                # 文生图：无参考图
                request_data = API_CLIENT.create_request_data(
                    model=模型,
                    prompt=提示词,
                    size=effective_aspect_ratio if effective_aspect_ratio != "Auto" else None,
                    n=生成数量,
                    response_format="b64_json",
                    image_size=图片尺寸,
                )

            response_data = API_CLIENT.send_request(
                resolved_api_key,
                request_data,
                api_base_url,
                timeout=(15, timeout_value),
            )

            base64_images, revised_prompt = API_CLIENT.extract_images(response_data)

            if not base64_images:
                error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", "未返回图片")
                return (error_tensor, "未返回图片")

            image_tensor = self.image_codec.base64_to_tensor_parallel(base64_images)

            total_time = time.time() - start_time
            info_text = f"✅ 生成 {len(base64_images)} 张图像 ({图片尺寸})，耗时 {total_time:.2f}s"

            logger.success(info_text)
            return (image_tensor, info_text)

        except Exception as e:
            error_msg = str(e)[:300]
            logger.error(f"生成失败: {error_msg}")
            error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
            return (error_tensor, error_msg)


class NebulaGeminiLiteNode:
    """
    Gemini Lite 图像生成节点
    专门用于 gemini-3.1-flash-lite-image 模型（固定 1K 分辨率，支持不同比例画布）
    其余功能与 NebulaGeminiNode 一致。
    """

    LITE_MODEL = "gemini-3.1-flash-lite-image"   # 模型固定
    LITE_SIZE = "1K"                              # 尺寸固定（该模型仅支持 1K）

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "信息")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/nebula图像节点"

    def __init__(self):
        self.config_manager = CONFIG_MANAGER
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影",
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "宽高比": (["Auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                    "default": "Auto",
                    "tooltip": "Auto 会自动匹配参考图像的相近宽高比"
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "超时秒数": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1800,
                    "tooltip": "API 请求超时时间（秒），0 表示无限等待"
                }),
                "参考图像1": ("IMAGE", {"tooltip": "参考图像 1"}),
                "参考图像2": ("IMAGE", {"tooltip": "参考图像 2"}),
                "参考图像3": ("IMAGE", {"tooltip": "参考图像 3"}),
                "参考图像4": ("IMAGE", {"tooltip": "参考图像 4"}),
                "参考图像5": ("IMAGE", {"tooltip": "参考图像 5"}),
                "参考图像6": ("IMAGE", {"tooltip": "参考图像 6"}),
                "参考图像7": ("IMAGE", {"tooltip": "参考图像 7"}),
                "参考图像8": ("IMAGE", {"tooltip": "参考图像 8"}),
                "参考图像9": ("IMAGE", {"tooltip": "参考图像 9"}),
                "参考图像10": ("IMAGE", {"tooltip": "参考图像 10"}),
                "参考图像11": ("IMAGE", {"tooltip": "参考图像 11"}),
                "参考图像12": ("IMAGE", {"tooltip": "参考图像 12"}),
                "参考图像13": ("IMAGE", {"tooltip": "参考图像 13"}),
                "参考图像14": ("IMAGE", {"tooltip": "参考图像 14"}),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def generate(
        self,
        提示词: str,
        API密钥: str = "",
        宽高比: str = "Auto",
        生成数量: int = 1,
        超时秒数: int = 0,
        **kwargs
    ):
        """生成 Gemini Lite 图像（固定 gemini-3.1-flash-lite-image / 1K）"""
        模型 = self.LITE_MODEL
        图片尺寸 = self.LITE_SIZE
        start_time = time.time()

        # 解析 API Key
        raw_api_key = (API密钥 or "").strip()
        resolved_api_key = self.config_manager.sanitize_api_key(raw_api_key)
        if not resolved_api_key:
            resolved_api_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_api_key:
            error_msg = "请配置 API Key"
            error_tensor = self.error_canvas.build_error_tensor_from_text("配置缺失", error_msg)
            return (error_tensor, error_msg)

        api_base_url = self.config_manager.get_effective_api_base_url()

        # 收集所有参考图像
        input_tensors = []
        for i in range(1, 15):
            img = kwargs.get(f"参考图像{i}")
            if img is not None:
                input_tensors.append(img)

        input_images_b64 = self.image_codec.prepare_input_images(input_tensors) if input_tensors else []

        # 自动检测宽高比
        effective_aspect_ratio = 宽高比
        if 宽高比 == "Auto" and input_tensors:
            # 从第一张参考图像获取宽高比
            first_img = input_tensors[0]
            if first_img is not None and len(first_img.shape) >= 3:
                h, w = first_img.shape[1], first_img.shape[2]
                ratio = w / h
                # 匹配最接近的宽高比
                aspect_ratios = {
                    "1:1": 1.0, "3:2": 1.5, "2:3": 0.667, "3:4": 0.75, "4:3": 1.333,
                    "4:5": 0.8, "5:4": 1.25, "9:16": 0.5625, "16:9": 1.778, "21:9": 2.333
                }
                closest = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
                effective_aspect_ratio = closest[0]
                logger.info(f"自动检测宽高比: {effective_aspect_ratio} (原始比例: {ratio:.2f})")

        logger.header("🌌 Gemini Lite 图像生成")
        logger.info(f"模型: {模型} (固定 1K)")
        logger.info(f"宽高比: {effective_aspect_ratio}")
        if input_tensors:
            logger.info(f"参考图像: {len(input_tensors)} 张")

        # 处理超时：0 表示无限等待
        timeout_value = None if 超时秒数 == 0 else 超时秒数

        try:
            self._ensure_not_interrupted()

            if input_images_b64:
                # 图生图：参考图放扁平的 image(单图)/ images(数组) 字段，
                # 不走 contents[].parts[] 格式（对方 demo 确认的标准用法）
                request_data = {
                    "model": 模型,
                    "prompt": 提示词,
                    "n": 生成数量,
                    "response_format": "b64_json",
                    "image_size": 图片尺寸,
                }
                if effective_aspect_ratio != "Auto":
                    request_data["size"] = effective_aspect_ratio
                if len(input_images_b64) == 1:
                    request_data["image"] = f"data:image/png;base64,{input_images_b64[0]}"
                else:
                    request_data["images"] = [
                        f"data:image/png;base64,{b64}" for b64 in input_images_b64
                    ]
            else:
                # 文生图：无参考图
                request_data = API_CLIENT.create_request_data(
                    model=模型,
                    prompt=提示词,
                    size=effective_aspect_ratio if effective_aspect_ratio != "Auto" else None,
                    n=生成数量,
                    response_format="b64_json",
                    image_size=图片尺寸,
                )

            response_data = API_CLIENT.send_request(
                resolved_api_key,
                request_data,
                api_base_url,
                timeout=(15, timeout_value),
            )

            base64_images, revised_prompt = API_CLIENT.extract_images(response_data)

            if not base64_images:
                error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", "未返回图片")
                return (error_tensor, "未返回图片")

            image_tensor = self.image_codec.base64_to_tensor_parallel(base64_images)

            total_time = time.time() - start_time
            info_text = f"✅ 生成 {len(base64_images)} 张图像 ({图片尺寸})，耗时 {total_time:.2f}s"

            logger.success(info_text)
            return (image_tensor, info_text)

        except Exception as e:
            error_msg = str(e)[:300]
            logger.error(f"生成失败: {error_msg}")
            error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
            return (error_tensor, error_msg)


class NebulaDoubaoNode:
    """
    豆包 Seedream 图像生成节点
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "信息")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/nebula图像节点"

    def __init__(self):
        self.config_manager = CONFIG_MANAGER
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)

    @classmethod
    def INPUT_TYPES(cls):
        # 豆包支持的尺寸
        sizes_3_0 = ["1024x1024", "1152x864", "864x1152", "1280x720", "720x1280", "1248x832", "832x1248", "1512x648"]
        sizes_4_x = ["2048x2048", "2304x1728", "1728x2304", "2560x1440", "1440x2560", "2496x1664", "1664x2496", "3024x1296"]
        all_sizes = sizes_3_0 + sizes_4_x

        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影",
                }),
                "模型": (DOUBAO_MODELS, {
                    "default": "doubao-seedream-4-0-250828",
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "图片尺寸": (all_sizes, {
                    "default": "2048x2048",
                }),
                "水印": ("BOOLEAN", {
                    "default": False,
                }),
                "随机种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                }),
                "引导系数": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "仅 3.0 模型支持"
                }),
                "优化模式": (["standard", "fast", "creative", "precise"], {
                    "default": "standard",
                    "tooltip": "4.0 支持 standard/fast，4.5 支持 standard/creative/precise"
                }),
                "参考图像": ("IMAGE", {}),
                "超时秒数": ("INT", {
                    "default": 420,
                    "min": 30,
                    "max": 1800,
                }),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def generate(
        self,
        提示词: str,
        模型: str,
        API密钥: str = "",
        图片尺寸: str = "2048x2048",
        水印: bool = False,
        随机种子: int = -1,
        引导系数: float = 2.5,
        优化模式: str = "standard",
        参考图像: Optional[torch.Tensor] = None,
        超时秒数: int = 420,
    ):
        """生成豆包图像"""
        start_time = time.time()

        raw_api_key = (API密钥 or "").strip()
        resolved_api_key = self.config_manager.sanitize_api_key(raw_api_key)
        if not resolved_api_key:
            resolved_api_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_api_key:
            error_msg = "请配置 API Key"
            error_tensor = self.error_canvas.build_error_tensor_from_text("配置缺失", error_msg)
            return (error_tensor, error_msg)

        api_base_url = self.config_manager.get_effective_api_base_url()

        input_images_b64 = []
        if 参考图像 is not None:
            input_images_b64 = self.image_codec.prepare_input_images([参考图像])

        # 构建额外参数
        extra_params = {"watermark": 水印}
        if 随机种子 >= 0:
            extra_params["seed"] = 随机种子
        if "3-0" in 模型:
            extra_params["guidance_scale"] = 引导系数
        if "4-0" in 模型 or "4-5" in 模型:
            extra_params["optimize_prompt_options"] = {"mode": 优化模式}

        logger.header("🌌 豆包 Seedream 图像生成")
        logger.info(f"模型: {模型}")
        logger.info(f"尺寸: {图片尺寸}")

        try:
            self._ensure_not_interrupted()

            request_data = API_CLIENT.create_request_data(
                model=模型,
                prompt=提示词,
                size=图片尺寸,
                n=1,
                response_format="url",
                input_images_b64=input_images_b64 if input_images_b64 else None,
                **extra_params
            )

            response_data = API_CLIENT.send_request(
                resolved_api_key,
                request_data,
                api_base_url,
                timeout=(15, 超时秒数),
            )

            base64_images, _ = API_CLIENT.extract_images(response_data)

            if not base64_images:
                error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", "未返回图片")
                return (error_tensor, "未返回图片")

            image_tensor = self.image_codec.base64_to_tensor_parallel(base64_images)

            total_time = time.time() - start_time
            info_text = f"✅ 生成 {len(base64_images)} 张图像，耗时 {total_time:.2f}s"

            logger.success(info_text)
            return (image_tensor, info_text)

        except Exception as e:
            error_msg = str(e)[:300]
            logger.error(f"生成失败: {error_msg}")
            error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
            return (error_tensor, error_msg)


class NebulaGPTImageNode:
    """
    GPT Image 图像生成节点
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "信息")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/nebula图像节点"

    def __init__(self):
        self.config_manager = CONFIG_MANAGER
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影",
                }),
                "模型": (GPT_MODELS, {
                    "default": "gpt-image-2",
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "图片尺寸": (["1024x1024", "1024x1536", "1536x1024"], {
                    "default": "1024x1024",
                }),
                "图片质量": (["low", "medium", "high"], {
                    "default": "high",
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }),
                "输入保真度": (["auto", "low", "medium", "high"], {
                    "default": "medium",
                }),
                "参考图像": ("IMAGE", {}),
                "超时秒数": ("INT", {
                    "default": 420,
                    "min": 30,
                    "max": 1800,
                }),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def generate(
        self,
        提示词: str,
        模型: str = "gpt-image-2",
        API密钥: str = "",
        图片尺寸: str = "1024x1024",
        图片质量: str = "high",
        生成数量: int = 1,
        输入保真度: str = "medium",
        参考图像: Optional[torch.Tensor] = None,
        超时秒数: int = 420,
    ):
        """生成 GPT 图像"""
        start_time = time.time()

        raw_api_key = (API密钥 or "").strip()
        resolved_api_key = self.config_manager.sanitize_api_key(raw_api_key)
        if not resolved_api_key:
            resolved_api_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_api_key:
            error_msg = "请配置 API Key"
            error_tensor = self.error_canvas.build_error_tensor_from_text("配置缺失", error_msg)
            return (error_tensor, error_msg)

        api_base_url = self.config_manager.get_effective_api_base_url()

        input_images_b64 = []
        if 参考图像 is not None:
            input_images_b64 = self.image_codec.prepare_input_images([参考图像])

        logger.header("🌌 GPT Image 图像生成")
        logger.info(f"模型: {模型}")
        logger.info(f"尺寸: {图片尺寸}")
        logger.info(f"质量: {图片质量}")
        if input_images_b64:
            logger.info(f"模式: 图生图（{len(input_images_b64)} 张参考图）")
        else:
            logger.info(f"模式: 文生图")

        try:
            self._ensure_not_interrupted()

            if input_images_b64:
                # 图生图：GPT 系列使用扁平的 image(单图)/ images(数组) 字段
                # （文档：图像生成.md:322, 337-340），不走 NebulaApiClient 的 contents 格式
                request_data = {
                    "model": 模型,
                    "prompt": 提示词,
                    "size": 图片尺寸,
                    "quality": 图片质量,
                    "n": 生成数量,
                    "response_format": "b64_json",
                    "input_fidelity": 输入保真度,
                }
                if len(input_images_b64) == 1:
                    request_data["image"] = f"data:image/png;base64,{input_images_b64[0]}"
                else:
                    request_data["images"] = [
                        f"data:image/png;base64,{b64}" for b64 in input_images_b64
                    ]
            else:
                # 文生图：input_fidelity 仅图生图有效，文生图不传（文档：图像生成.md:459）
                request_data = API_CLIENT.create_request_data(
                    model=模型,
                    prompt=提示词,
                    size=图片尺寸,
                    quality=图片质量,
                    n=生成数量,
                    response_format="b64_json",
                )

            response_data = API_CLIENT.send_request(
                resolved_api_key,
                request_data,
                api_base_url,
                timeout=(15, 超时秒数),
            )

            base64_images, _ = API_CLIENT.extract_images(response_data)

            if not base64_images:
                error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", "未返回图片")
                return (error_tensor, "未返回图片")

            image_tensor = self.image_codec.base64_to_tensor_parallel(base64_images)

            total_time = time.time() - start_time
            info_text = f"✅ 生成 {len(base64_images)} 张图像，耗时 {total_time:.2f}s"

            logger.success(info_text)
            return (image_tensor, info_text)

        except Exception as e:
            error_msg = str(e)[:300]
            logger.error(f"生成失败: {error_msg}")
            error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
            return (error_tensor, error_msg)


class NebulaQwenImageNode:
    """
    通义千问图像生成节点
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "信息")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/nebula图像节点"

    def __init__(self):
        self.config_manager = CONFIG_MANAGER
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的橙色小猫坐在花园里，阳光明媚，高质量摄影",
                }),
                "模型": (QWEN_MODELS, {
                    "default": "qwen-image-plus",
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "图片尺寸": (["1328*1328", "1664*928", "928*1664", "1472*1140", "1140*1472"], {
                    "default": "1328*1328",
                }),
                "负面提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "提示词扩展": ("BOOLEAN", {
                    "default": True,
                }),
                "水印": ("BOOLEAN", {
                    "default": False,
                }),
                "参考图像": ("IMAGE", {
                    "tooltip": "仅 qwen-image-edit-plus 支持"
                }),
                "超时秒数": ("INT", {
                    "default": 420,
                    "min": 30,
                    "max": 1800,
                }),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def generate(
        self,
        提示词: str,
        模型: str,
        API密钥: str = "",
        图片尺寸: str = "1328*1328",
        负面提示词: str = "",
        提示词扩展: bool = True,
        水印: bool = False,
        参考图像: Optional[torch.Tensor] = None,
        超时秒数: int = 420,
    ):
        """生成通义千问图像"""
        start_time = time.time()

        raw_api_key = (API密钥 or "").strip()
        resolved_api_key = self.config_manager.sanitize_api_key(raw_api_key)
        if not resolved_api_key:
            resolved_api_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_api_key:
            error_msg = "请配置 API Key"
            error_tensor = self.error_canvas.build_error_tensor_from_text("配置缺失", error_msg)
            return (error_tensor, error_msg)

        api_base_url = self.config_manager.get_effective_api_base_url()

        input_images_b64 = []
        if 参考图像 is not None and "edit" in 模型:
            input_images_b64 = self.image_codec.prepare_input_images([参考图像])

        # 通义千问特殊参数格式
        parameters = {
            "size": 图片尺寸,
            "prompt_extend": 提示词扩展,
            "watermark": 水印,
        }
        if 负面提示词:
            parameters["negative_prompt"] = 负面提示词

        logger.header("🌌 通义千问图像生成")
        logger.info(f"模型: {模型}")
        logger.info(f"尺寸: {图片尺寸}")

        try:
            self._ensure_not_interrupted()

            request_data = API_CLIENT.create_request_data(
                model=模型,
                prompt=提示词,
                n=1,
                response_format="b64_json",
                input_images_b64=input_images_b64 if input_images_b64 else None,
                parameters=parameters
            )

            response_data = API_CLIENT.send_request(
                resolved_api_key,
                request_data,
                api_base_url,
                timeout=(15, 超时秒数),
            )

            base64_images, _ = API_CLIENT.extract_images(response_data)

            if not base64_images:
                error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", "未返回图片")
                return (error_tensor, "未返回图片")

            image_tensor = self.image_codec.base64_to_tensor_parallel(base64_images)

            total_time = time.time() - start_time
            info_text = f"✅ 生成 {len(base64_images)} 张图像，耗时 {total_time:.2f}s"

            logger.success(info_text)
            return (image_tensor, info_text)

        except Exception as e:
            error_msg = str(e)[:300]
            logger.error(f"生成失败: {error_msg}")
            error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
            return (error_tensor, error_msg)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFOLIE_NebulaImageGenerator": NebulaImageGenerator,
    "AFOLIE_NebulaGemini": NebulaGeminiNode,
    "AFOLIE_NebulaGeminiLite": NebulaGeminiLiteNode,
    "AFOLIE_NebulaDoubao": NebulaDoubaoNode,
    "AFOLIE_NebulaGPTImage": NebulaGPTImageNode,
    "AFOLIE_NebulaQwenImage": NebulaQwenImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFOLIE_NebulaImageGenerator": "🌌 Nebula 图像生成",
    "AFOLIE_NebulaGemini": "🌌 Nebula Gemini",
    "AFOLIE_NebulaGeminiLite": "🌌 Nebula Gemini-lite",
    "AFOLIE_NebulaDoubao": "🌌 Nebula 豆包 Seedream",
    "AFOLIE_NebulaGPTImage": "🌌 Nebula GPT Image",
    "AFOLIE_NebulaQwenImage": "🌌 Nebula 通义千问",
}
