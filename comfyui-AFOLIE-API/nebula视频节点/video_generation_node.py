"""
Nebula 视频生成节点
支持多种模型：Sora 2, Veo, 阿里万相, 豆包 Seedance
"""

from __future__ import annotations

import base64
import os
import sys
import time
import io
import tempfile
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
import requests

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(MODULE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from server import PromptServer
except ImportError:
    class _DummyPromptServer:
        instance = None
    PromptServer = _DummyPromptServer()

import comfy.utils
import comfy.model_management

try:
    from nebula_logger import logger
except ImportError:
    logger = None

try:
    from nebula_config_manager import ConfigManager
except ImportError:
    ConfigManager = None

from nebula_video_client import VideoClient, VideoModel


class VideoGenerationNode:
    """
    Nebula 视频生成节点
    支持文生视频、图生视频
    """

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "info")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/视频节点"

    # 可用模型列表
    MODELS = [
        "sora-2",
        "veo-3.0-fast-generate-001",
        "veo-3.1-fast-generate-preview",
        "wan2.5-t2v-preview",
        "wan2.5-i2v-preview",
        "doubao-seedance-1-0-lite-t2v-250428",
        "doubao-seedance-1-0-lite-i2v-250428",
        "doubao-seedance-1-0-pro-250528",
    ]

    def __init__(self):
        self.interrupt_checker = comfy.model_management.throw_exception_if_processing_interrupted
        if ConfigManager:
            self.config_manager = ConfigManager(PARENT_DIR)
        else:
            self.config_manager = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的小猫在花园里玩耍，阳光明媚，画面温馨",
                    "tooltip": "视频生成提示词"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API 密钥（可选，可用 config.ini 配置）"
                }),
                "model": (cls.MODELS, {
                    "default": "sora-2",
                    "tooltip": "选择视频生成模型"
                }),
                "任务模式": (["仅提交", "提交并等待完成"], {
                    "default": "提交并等待完成",
                    "tooltip": "仅提交：返回任务ID，可用于后续查询；提交并等待完成：等待视频生成完成"
                }),
            },
            "optional": {
                "输入图像": ("IMAGE", {
                    "tooltip": "参考图像（图生视频模式）"
                }),
                "尾帧图像": ("IMAGE", {
                    "tooltip": "尾帧图像（仅 Veo 3.1 和豆包支持）"
                }),
                "视频时长": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "tooltip": "视频时长（秒），根据模型不同支持范围不同"
                }),
                "分辨率": (["720p", "480p", "1080p", "4k"], {
                    "default": "720p",
                    "tooltip": "视频分辨率"
                }),
                "宽高比": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"], {
                    "default": "16:9",
                    "tooltip": "视频宽高比，adaptive 仅部分模型支持"
                }),
                "帧率": ("INT", {
                    "default": 24,
                    "min": 12,
                    "max": 60,
                    "tooltip": "帧率（fps）"
                }),
                "随机种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "随机种子（-1 为随机，固定值可复现）"
                }),
                "生成音频": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否生成同步音频"
                }),
                "添加水印": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
                "人像生成": (["allow_all", "allow_adult", "dont_allow"], {
                    "default": "allow_all",
                    "tooltip": "人像生成策略（仅 Veo）"
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "tooltip": "每次生成的视频数量（仅 Veo）"
                }),
                "Remix视频ID": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "基于已有视频重新生成（仅 Sora 2 Remix 模式，需以 video_ 开头）"
                }),
                "豆包提示词": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "豆包专用提示词格式（支持 --ratio --dur --rs 等参数）"
                }),
                "轮询间隔": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 1.0,
                    "tooltip": "轮询查询间隔（秒）"
                }),
                "最大等待时间": ("INT", {
                    "default": 3600,
                    "min": 60,
                    "max": 7200,
                    "tooltip": "最大等待时间（秒）"
                }),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def _resolve_api_key(self, provided_key: str) -> str:
        """解析 API Key"""
        raw_key = (provided_key or "").strip()
        if not raw_key and self.config_manager:
            resolved_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )
            if not resolved_key:
                raise ValueError("请配置有效的 API Key（在 config.ini 或节点输入中）")
            return resolved_key
        elif raw_key:
            if self.config_manager:
                raw_key = self.config_manager.sanitize_api_key(raw_key)
            return raw_key
        else:
            raise ValueError("请配置有效的 API Key（在 config.ini 或节点输入中）")

    def _image_to_base64(self, image_tensor: torch.Tensor) -> str:
        """将图像张量转换为 Base64 编码"""
        if image_tensor is None:
            return ""

        # 转换为 numpy 数组
        image_np = image_tensor.cpu().numpy()

        # 确保形状正确 [B, H, W, C]
        if len(image_np.shape) == 3:
            image_np = np.expand_dims(image_np, 0)

        # 取第一张图
        image_np = image_np[0]

        # 转换为 uint8
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        # 转换为 RGB
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]

        # 转换为 bytes 并编码为 base64
        import io
        from PIL import Image

        pil_image = Image.fromarray(image_np)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')

        return f"data:image/png;base64,{base64_str}"

    def _get_api_base_url(self) -> str:
        """获取 API Base URL"""
        if self.config_manager:
            return self.config_manager.get_effective_api_base_url()
        return "https://llm.ai-nebula.com"

    def _build_doubao_metadata(
        self,
        prompt: str,
        first_frame: Optional[str] = None,
        last_frame: Optional[str] = None,
        reference_images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """构建豆包 Seedance 的 metadata 格式"""
        content = []

        # 添加文本提示词
        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })

        # 添加首帧
        if first_frame:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": first_frame
                },
                "role": "first_frame"
            })

        # 添加尾帧
        if last_frame:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": last_frame
                },
                "role": "last_frame"
            })

        # 添加参考图
        if reference_images:
            for ref_img in reference_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": ref_img
                    },
                    "role": "reference_image"
                })

        return {"content": content}

    def generate(
        self,
        prompt: str,
        api_key: str,
        model: str,
        任务模式: str = "提交并等待完成",
        输入图像: Optional[torch.Tensor] = None,
        尾帧图像: Optional[torch.Tensor] = None,
        视频时长: int = 5,
        分辨率: str = "720p",
        宽高比: str = "16:9",
        帧率: int = 24,
        随机种子: int = -1,
        生成音频: bool = False,
        添加水印: bool = False,
        人像生成: str = "allow_all",
        生成数量: int = 1,
        Remix视频ID: str = "",
        豆包提示词: str = "",
        轮询间隔: float = 5.0,
        最大等待时间: int = 3600,
    ) -> Tuple[Dict[str, Any], str]:
        """生成视频"""
        start_time = time.time()

        try:
            self._ensure_not_interrupted()

            # 解析 API Key
            resolved_api_key = self._resolve_api_key(api_key)

            # 获取 API Base URL
            api_base_url = self._get_api_base_url()

            # 创建客户端
            client = VideoClient(
                api_base_url=api_base_url,
                api_key=resolved_api_key,
                interrupt_checker=self._ensure_not_interrupted
            )

            # 准备图像
            image_base64 = None
            if 输入图像 is not None:
                image_base64 = self._image_to_base64(输入图像)

            last_frame_base64 = None
            if 尾帧图像 is not None:
                last_frame_base64 = self._image_to_base64(尾帧图像)

            # 构建请求参数
            request_params = {
                "model": model,
            }

            # 处理豆包模型的特殊格式
            if model.startswith("doubao"):
                # 使用豆包专用提示词，如果没有则使用普通提示词
                effective_prompt = 豆包提示词 if 豆包提示词 else prompt

                # 构建豆包 metadata
                metadata = self._build_doubao_metadata(
                    prompt=effective_prompt,
                    first_frame=image_base64,
                    last_frame=last_frame_base64
                )
                request_params["metadata"] = metadata
            else:
                # 其他模型使用标准 prompt
                request_params["prompt"] = prompt
                if image_base64:
                    request_params["image"] = image_base64

            # 添加通用参数
            if 随机种子 >= 0:
                request_params["seed"] = 随机种子

            # Sora 2 专用参数
            if model == "sora-2":
                size_map = {
                    "16:9": "1280x720",
                    "9:16": "720x1280",
                }
                request_params["seconds"] = str(视频时长)
                request_params["size"] = size_map.get(宽高比, "1280x720")
                if image_base64:
                    request_params["input_reference"] = image_base64
                if Remix视频ID and Remix视频ID.startswith("video_"):
                    request_params["remix_from_video_id"] = Remix视频ID

            # Veo 专用参数
            elif model.startswith("veo"):
                request_params["duration_seconds"] = 视频时长
                request_params["aspect_ratio"] = 宽高比
                request_params["resolution"] = 分辨率
                request_params["fps"] = 帧率
                request_params["generate_audio"] = 生成音频
                request_params["person_generation"] = 人像生成
                request_params["add_watermark"] = 添加水印
                request_params["sample_count"] = 生成数量
                if last_frame_base64 and "veo-3.1" in model:
                    request_params["last_frame"] = last_frame_base64

            # 阿里万相专用参数
            elif model.startswith("wan"):
                request_params["duration"] = 视频时长
                request_params["resolution"] = 分辨率
                if model == "wan2.5-t2v-preview":
                    size_map = {
                        "16:9": "1280*720",
                        "9:16": "720*1280",
                        "1:1": "1080*1080",
                    }
                    request_params["size"] = size_map.get(宽高比, "1280*720")

            # 记录日志
            if logger:
                logger.header("🎬 Nebula 视频生成任务")
                logger.info(f"模型: {model}")
                logger.info(f"任务模式: {任务模式}")
                logger.info(f"视频时长: {视频时长}秒")
                logger.info(f"分辨率: {分辨率}")
                logger.info(f"宽高比: {宽高比}")
                if 输入图像 is not None:
                    logger.info("图生视频模式: 启用")
                logger.separator()

            # 提交任务
            submit_result = client.submit_video_task(**request_params)
            task_id = submit_result.get("task_id", "")

            if not task_id:
                raise RuntimeError("未返回任务 ID")

            submit_time = time.time() - start_time

            # 仅提交模式
            if 任务模式 == "仅提交":
                info_text = f"✅ 任务已提交\n"
                info_text += f"任务 ID: {task_id}\n"
                info_text += f"模型: {model}\n"
                info_text += f"提交耗时: {submit_time:.2f}s\n\n"
                info_text += f"💡 请使用'查询视频任务'节点查询任务状态"

                return (None, info_text)

            # 等待完成模式
            if logger:
                logger.info(f"任务 ID: {task_id}")
                logger.info(f"开始轮询等待任务完成...")

            # 进度回调函数
            def progress_callback(result: Dict[str, Any], elapsed: float):
                status = result.get("status", "unknown")
                if logger and status not in ["submitted", "processing"]:
                    logger.info(f"任务状态: {status} (已等待 {elapsed:.0f}s)")

            # 等待任务完成
            result = client.wait_for_task_completion(
                task_id=task_id,
                poll_interval=轮询间隔,
                max_wait_time=最大等待时间,
                progress_callback=progress_callback
            )

            total_time = time.time() - start_time

            # 提取视频 URL
            video_url = ""
            if "video" in result:
                video_data = result["video"]
                if isinstance(video_data, dict) and "url" in video_data:
                    video_url = video_data["url"]
                elif isinstance(video_data, str):
                    video_url = video_data

            # 下载视频
            video_path = ""
            if video_url:
                if logger:
                    logger.info(f"正在下载视频: {video_url[:80]}...")
                
                self._ensure_not_interrupted()
                
                try:
                    response = requests.get(video_url, timeout=120, verify=True)
                    response.raise_for_status()
                    
                    video_bytes = response.content
                    
                    if logger:
                        logger.info(f"视频下载成功: {len(video_bytes)} 字节")
                    
                    # 保存到临时文件并返回文件路径
                    temp_dir = tempfile.gettempdir()
                    video_filename = f"nebula_{task_id}.mp4"
                    video_path = os.path.join(temp_dir, video_filename)
                    
                    with open(video_path, 'wb') as f:
                        f.write(video_bytes)
                    
                    if logger:
                        logger.info(f"视频已保存: {video_path}")
                except Exception as e:
                    if logger:
                        logger.error(f"视频下载失败: {str(e)}")
                    info_text += f"\n⚠️ 视频下载失败: {str(e)}"

            # 构建返回信息
            info_text = f"✅ 视频生成完成\n"
            info_text += f"任务 ID: {task_id}\n"
            info_text += f"模型: {model}\n"
            info_text += f"视频时长: {视频时长}秒\n"
            info_text += f"分辨率: {分辨率}\n"
            info_text += f"宽高比: {宽高比}\n"
            info_text += f"总耗时: {total_time:.2f}s\n"
            if video_url:
                info_text += f"视频链接: {video_url[:100]}..."

            if logger:
                logger.summary("任务完成", {
                    "任务ID": task_id,
                    "总耗时": f"{total_time:.2f}s",
                    "视频": "已下载" if video_path else "未下载"
                })

            return (video_path or None, info_text)

        except comfy.model_management.InterruptProcessingException:
            if logger:
                logger.warning("任务已取消")
            raise
        except Exception as e:
            error_msg = str(e)[:500]
            if logger:
                logger.error(f"生成失败: {error_msg}")
            info_text = f"❌ 视频生成失败\n\n错误信息: {error_msg}"
            return (None, info_text)


class VideoQueryNode:
    """
    Nebula 视频任务查询节点
    """

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_id", "info")
    FUNCTION = "query"
    OUTPUT_NODE = False
    CATEGORY = "AFOLIE/API/视频节点"

    def __init__(self):
        self.interrupt_checker = comfy.model_management.throw_exception_if_processing_interrupted
        if ConfigManager:
            self.config_manager = ConfigManager(PARENT_DIR)
        else:
            self.config_manager = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "任务 ID（以 video_ 开头）"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API 密钥（可选，可用 config.ini 配置）"
                }),
            },
        }

    def _resolve_api_key(self, provided_key: str) -> str:
        """解析 API Key"""
        raw_key = (provided_key or "").strip()
        if not raw_key and self.config_manager:
            resolved_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )
            if not resolved_key:
                raise ValueError("请配置有效的 API Key（在 config.ini 或节点输入中）")
            return resolved_key
        elif raw_key:
            if self.config_manager:
                raw_key = self.config_manager.sanitize_api_key(raw_key)
            return raw_key
        else:
            raise ValueError("请配置有效的 API Key（在 config.ini 或节点输入中）")

    def _get_api_base_url(self) -> str:
        """获取 API Base URL"""
        if self.config_manager:
            return self.config_manager.get_effective_api_base_url()
        return "https://llm.ai-nebula.com"

    def query(
        self,
        task_id: str,
        api_key: str,
    ) -> Tuple[str, str]:
        """查询任务状态"""
        try:
            self._ensure_not_interrupted()

            # 解析 API Key
            resolved_api_key = self._resolve_api_key(api_key)

            # 获取 API Base URL
            api_base_url = self._get_api_base_url()

            # 创建客户端
            client = VideoClient(
                api_base_url=api_base_url,
                api_key=resolved_api_key,
                interrupt_checker=self._ensure_not_interrupted
            )

            # 查询任务
            result = client.query_video_task(task_id)

            # 构建返回信息
            status = result.get("status", "unknown")
            info_text = f"任务 ID: {task_id}\n"
            info_text += f"状态: {status}\n"

            # 提取视频 URL
            video_url = ""
            if "video" in result:
                video_data = result["video"]
                if isinstance(video_data, dict) and "url" in video_data:
                    video_url = video_data["url"]
                elif isinstance(video_data, str):
                    video_url = video_data
                if video_url:
                    info_text += f"视频链接: {video_url[:100]}..."

            if "error" in result:
                info_text += f"\n错误信息: {result['error']}"

            return (task_id, info_text)

        except Exception as e:
            error_msg = str(e)[:500]
            info_text = f"❌ 查询失败\n\n错误信息: {error_msg}"
            return (task_id, info_text)

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFOLIE_VideoGeneration": VideoGenerationNode,
    "AFOLIE_VideoQuery": VideoQueryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFOLIE_VideoGeneration": "🎬 AFOLIE Nebula 视频生成",
    "AFOLIE_VideoQuery": "🔍 AFOLIE 查询视频任务",
}
