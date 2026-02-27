"""
高级图像生成节点
支持 Gemini 和 OpenAI 双协议
批量生成、多图参考、智能参数处理
"""

from __future__ import annotations

import base64
import json
import time
import os
import sys
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import requests

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


class GeminiApiClient:
    """Gemini 协议 API 客户端"""

    def __init__(self, config_manager: ConfigManager, logger_instance=logger, interrupt_checker=None):
        self.config_manager = config_manager
        self.logger = logger_instance
        self.interrupt_checker = interrupt_checker

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _ensure_not_interrupted(self):
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    def create_request_data(
        self,
        model: str,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        input_images_b64: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        image_size: Optional[str] = None,
        **extra_params
    ) -> Dict[str, Any]:
        """创建 Gemini 请求格式"""
        request_body: Dict[str, Any] = {
            "model": model,
            "n": n,
            "response_format": "b64_json",
        }

        # 处理提示词和参考图像
        if input_images_b64:
            parts = []
            for img_b64 in input_images_b64:
                parts.append({"image": f"data:image/png;base64,{img_b64}"})
            if prompt:
                parts.append({"text": prompt})
            request_body["contents"] = [{
                "role": "user",
                "parts": parts
            }]
        else:
            request_body["prompt"] = prompt

        # 宽高比 - 使用 size 参数
        if size and size != "Auto":
            request_body["size"] = size
        
        # 质量 - 映射到 image_size（分辨率）
        # quality: "high"/"2K" -> image_size: "2K"
        # quality: "standard"/"medium"/"low"/"auto"/"1K" -> image_size: "1K"
        if quality:
            if quality in ["high", "2K"]:
                request_body["image_size"] = "2K"
            elif quality in ["standard", "medium", "low", "auto", "1K"]:
                request_body["image_size"] = "1K"
        
        # 直接使用 image_size 参数（如果提供）
        if image_size and image_size != "无":
            request_body["image_size"] = image_size

        # 可选参数
        if top_p is not None:
            request_body["top_p"] = top_p
        if seed is not None and seed >= 0:
            request_body["seed"] = seed

        # 额外参数
        for key, value in extra_params.items():
            if value is not None:
                request_body[key] = value

        return request_body

    def send_request(
        self,
        api_key: str,
        request_data: Dict[str, Any],
        api_base_url: str,
        timeout: Optional[Tuple[float, float]] = None,
        bypass_proxy: bool = False,
        verify_ssl: bool = True,
    ) -> Dict[str, Any]:
        """发送 API 请求"""
        sanitized_key = self.config_manager.sanitize_api_key(api_key)
        if not sanitized_key:
            raise ValueError("API Key 无效")

        url = f"{api_base_url.rstrip('/')}/images/generations"

        session = requests.Session()
        if bypass_proxy:
            session.trust_env = False
            session.proxies = {}

        headers = self._build_headers(sanitized_key)
        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")

        if timeout is None:
            timeout = (15, 420)

        try:
            self._ensure_not_interrupted()

            # 中断检查循环
            done_event = threading.Event()
            resp_holder = {}
            exc_holder = {}

            def _do_request():
                try:
                    resp_holder["resp"] = session.post(
                        url,
                        data=payload,
                        headers=headers,
                        timeout=timeout,
                        verify=verify_ssl,
                    )
                except BaseException as exc:
                    exc_holder["exc"] = exc
                finally:
                    done_event.set()

            thread = threading.Thread(target=_do_request, daemon=True)
            thread.start()

            poll_interval = 0.25
            while not done_event.wait(timeout=poll_interval):
                self._ensure_not_interrupted()
            self._ensure_not_interrupted()

            if "exc" in exc_holder:
                raise exc_holder["exc"]

            resp = resp_holder.get("resp")
            if resp is None:
                raise RuntimeError("请求被中断")

            resp.raise_for_status()
            return resp.json()

        except requests.HTTPError as e:
            error_msg = f"HTTP 错误: {e.response.status_code}"
            try:
                error_text = e.response.text[:500]
                error_msg += f"\n响应内容: {error_text}"
                
                # 尝试解析 JSON 错误
                try:
                    error_json = e.response.json()
                    if 'error' in error_json:
                        error_msg += f"\n错误详情: {error_json['error']}"
                except:
                    pass
            except Exception as parse_error:
                error_msg += f"\n无法解析响应: {parse_error}"
            raise RuntimeError(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"请求失败: {type(e).__name__} - {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_text = e.response.text[:500]
                    error_msg += f"\n响应内容: {error_text}"
                except:
                    pass
            raise RuntimeError(error_msg)


class OpenAIApiClient:
    """OpenAI 协议 API 客户端"""

    def __init__(self, config_manager: ConfigManager, logger_instance=logger, interrupt_checker=None):
        self.config_manager = config_manager
        self.logger = logger_instance
        self.interrupt_checker = interrupt_checker

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _ensure_not_interrupted(self):
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    def create_request_data(
        self,
        model: str,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        response_format: str = "b64_json",
        input_images_b64: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **extra_params
    ) -> Dict[str, Any]:
        """创建 OpenAI 请求格式"""
        request_body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "response_format": response_format,
        }

        if size:
            request_body["size"] = size
        if quality:
            request_body["quality"] = quality

        # 可选参数
        if top_p is not None:
            request_body["top_p"] = top_p
        if seed is not None and seed >= 0:
            request_body["seed"] = seed

        # 额外参数
        for key, value in extra_params.items():
            if value is not None:
                request_body[key] = value

        return request_body

    def send_request(
        self,
        api_key: str,
        request_data: Dict[str, Any],
        api_base_url: str,
        timeout: Optional[Tuple[float, float]] = None,
        bypass_proxy: bool = False,
        verify_ssl: bool = True,
    ) -> Dict[str, Any]:
        """发送 API 请求"""
        sanitized_key = self.config_manager.sanitize_api_key(api_key)
        if not sanitized_key:
            raise ValueError("API Key 无效")

        url = f"{api_base_url.rstrip('/')}/images/generations"

        session = requests.Session()
        if bypass_proxy:
            session.trust_env = False
            session.proxies = {}

        headers = self._build_headers(sanitized_key)
        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")

        if timeout is None:
            timeout = (15, 420)

        try:
            self._ensure_not_interrupted()

            done_event = threading.Event()
            resp_holder = {}
            exc_holder = {}

            def _do_request():
                try:
                    resp_holder["resp"] = session.post(
                        url,
                        data=payload,
                        headers=headers,
                        timeout=timeout,
                        verify=verify_ssl,
                    )
                except BaseException as exc:
                    exc_holder["exc"] = exc
                finally:
                    done_event.set()

            thread = threading.Thread(target=_do_request, daemon=True)
            thread.start()

            poll_interval = 0.25
            while not done_event.wait(timeout=poll_interval):
                self._ensure_not_interrupted()
            self._ensure_not_interrupted()

            if "exc" in exc_holder:
                raise exc_holder["exc"]

            resp = resp_holder.get("resp")
            if resp is None:
                raise RuntimeError("请求被中断")

            resp.raise_for_status()
            return resp.json()

        except requests.HTTPError as e:
            error_msg = f"HTTP 错误: {e.response.status_code}"
            try:
                error_text = e.response.text[:500]
                error_msg += f"\n响应内容: {error_text}"
                
                # 尝试解析 JSON 错误
                try:
                    error_json = e.response.json()
                    if 'error' in error_json:
                        error_msg += f"\n错误详情: {error_json['error']}"
                except:
                    pass
            except Exception as parse_error:
                error_msg += f"\n无法解析响应: {parse_error}"
            raise RuntimeError(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"请求失败: {type(e).__name__} - {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_text = e.response.text[:500]
                    error_msg += f"\n响应内容: {error_text}"
                except:
                    pass
            raise RuntimeError(error_msg)


class BatchGenerationRunner:
    """批量生成运行器"""

    def __init__(
        self,
        api_client,
        max_workers: int = 4,
        logger_instance=logger,
        interrupt_checker=None
    ):
        self.api_client = api_client
        self.max_workers = max_workers
        self.logger = logger_instance
        self.interrupt_checker = interrupt_checker

    def _ensure_not_interrupted(self):
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    def run_batch(
        self,
        api_key: str,
        request_data_template: Dict[str, Any],
        api_base_url: str,
        batch_size: int,
        timeout: Optional[Tuple[float, float]] = None,
        bypass_proxy: bool = False,
        verify_ssl: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """运行批量生成"""
        self._ensure_not_interrupted()

        results = []
        errors = []

        def _process_batch(batch_num: int, n: int) -> Optional[Dict[str, Any]]:
            """处理单个批次"""
            try:
                self._ensure_not_interrupted()

                # 创建该批次的请求数据
                batch_data = request_data_template.copy()
                batch_data["n"] = n

                # 发送请求
                response_data = self.api_client.send_request(
                    api_key=api_key,
                    request_data=batch_data,
                    api_base_url=api_base_url,
                    timeout=timeout,
                    bypass_proxy=bypass_proxy,
                    verify_ssl=verify_ssl,
                )

                return {
                    "batch": batch_num,
                    "count": n,
                    "response": response_data,
                }

            except Exception as e:
                error_msg = f"批次 {batch_num} 失败: {str(e)}"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                return None

        # 使用线程池执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            remaining = batch_size
            batch_num = 0

            while remaining > 0:
                self._ensure_not_interrupted()
                batch_num += 1
                # 每个批次生成 1 张图片
                n = 1

                future = executor.submit(_process_batch, batch_num, n)
                futures[future] = batch_num
                remaining -= n

            # 收集结果
            for future in as_completed(futures):
                self._ensure_not_interrupted()
                result = future.result()
                if result:
                    results.append(result)

        # 按批次号排序
        results.sort(key=lambda x: x["batch"])
        return results, errors


class ImageGenerationNode:
    """
    高级图像生成节点
    支持 Gemini/OpenAI 双协议、批量生成、多图参考
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "AFOLIE/API/高级图像节点"

    # 可用模型列表（仅 NebulaAI API 实际支持的模型）
    MODELS = [
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
    ]

    def __init__(self):
        self.config_manager = ConfigManager(MODULE_DIR)
        self.image_codec = ImageCodec(logger, self._ensure_not_interrupted)
        self.error_canvas = ErrorCanvas(logger)
        self.interrupt_checker = comfy.model_management.throw_exception_if_processing_interrupted

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful sunset over ocean, high quality, detailed",
                    "tooltip": "文本提示词，支持多行输入"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API 密钥（可选，可用 config.ini 配置）"
                }),
                "model_type": (cls.MODELS, {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "模型名称"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "tooltip": "批次大小（1-8），一次生成图片数量"
                }),
                "aspect_ratio": (["Auto", "1:1", "16:9", "9:16", "2:3", "3:2", "4:3", "3:4", "4:5", "5:4", "21:9"], {
                    "default": "Auto",
                    "tooltip": "宽高比：Auto/1:1/16:9/9:16/2:3/3:2/4:3/3:4/4:5/5:4/21:9"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "随机种子（-1 为随机，固定值可复现）"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "采样参数（0.0-1.0），控制多样性"
                }),
                "imageSize": (["无", "1K", "2K", "4K"], {
                    "default": "1K",
                    "tooltip": "分辨率：无/1K/2K/4K"
                }),
                "image_1": ("IMAGE", {"tooltip": "参考图像 1"}),
                "image_2": ("IMAGE", {"tooltip": "参考图像 2"}),
                "image_3": ("IMAGE", {"tooltip": "参考图像 3"}),
                "image_4": ("IMAGE", {"tooltip": "参考图像 4"}),
                "image_5": ("IMAGE", {"tooltip": "参考图像 5"}),
                "image_6": ("IMAGE", {"tooltip": "参考图像 6"}),
                "image_7": ("IMAGE", {"tooltip": "参考图像 7"}),
                "image_8": ("IMAGE", {"tooltip": "参考图像 8"}),
                "image_9": ("IMAGE", {"tooltip": "参考图像 9"}),
                "超时秒数": ("INT", {
                    "default": 1200,
                    "min": 0,
                    "max": 1800,
                    "tooltip": "API 请求超时时间（0-1800秒）"
                }),
                "绕过代理": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "梯子不稳定时开启"
                }),
                "api_protocol": (["Gemini", "OpenAI"], {
                    "default": "Gemini",
                    "tooltip": "选择 Gemini 或 OpenAI 协议"
                }),
            }
        }

    @staticmethod
    def _ensure_not_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()

    def _resolve_api_key(self, provided_key: str) -> str:
        """解析 API Key"""
        raw_key = (provided_key or "").strip()
        resolved_key = self.config_manager.sanitize_api_key(raw_key)
        if not resolved_key:
            resolved_key = self.config_manager.sanitize_api_key(
                self.config_manager.load_api_key()
            )

        if not resolved_key:
            raise ValueError("请配置有效的 API Key（在 config.ini 或节点输入中）")

        return resolved_key

    def _get_api_base_url(self) -> str:
        """获取 API Base URL（从配置文件读取）"""
        return self.config_manager.get_effective_api_base_url()

    def _collect_input_images(self, **kwargs) -> List[torch.Tensor]:
        """收集所有输入图像"""
        images = []
        for i in range(1, 10):
            img_key = f"image_{i}"
            img = kwargs.get(img_key)
            if img is not None:
                images.append(img)
        return images

    def _extract_images_from_response(
        self,
        response_data: Dict[str, Any],
        protocol: str
    ) -> List[str]:
        """从响应中提取图片"""
        images: List[str] = []

        if "data" in response_data:
            data = response_data["data"]

            # 处理嵌套的 data 结构
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # b64_json 格式
                        if "b64_json" in item and item["b64_json"]:
                            images.append(item["b64_json"])
                        # url 格式
                        elif "url" in item and item["url"]:
                            b64_data = self._download_image_to_base64(item["url"])
                            if b64_data:
                                images.append(b64_data)

        return images

    def _download_image_to_base64(self, url: str, timeout: float = 30.0) -> Optional[str]:
        """下载图片并转换为 base64"""
        try:
            session = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "image/*,*/*;q=0.8",
            }
            response = session.get(url, headers=headers, timeout=timeout, verify=True)
            response.raise_for_status()

            image_data = response.content
            base64_data = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"图片下载成功：{len(image_data)} 字节")
            return base64_data
        except Exception as exc:
            logger.warning(f"图片下载失败：{type(exc).__name__}")
            return None

    def generate(
        self,
        prompt: str,
        api_key: str = "",
        api_base_url: str = "",
        model_type: str = "gemini-3.1-flash-image-preview",
        batch_size: int = 1,
        aspect_ratio: str = "Auto",
        seed: int = -1,
        top_p: float = 1.0,
        imageSize: str = "无",
        超时秒数: int = 420,
        绕过代理: bool = False,
        api_protocol: str = "Gemini",
        **kwargs
    ):
        """生成图像"""
        start_time = time.time()

        try:
            self._ensure_not_interrupted()

            # 解析 API Key
            resolved_api_key = self._resolve_api_key(api_key)

            # 获取 API Base URL（从配置文件读取）
            resolved_api_base_url = self._get_api_base_url()

            # 收集输入图像
            input_tensors = self._collect_input_images(**kwargs)
            input_images_b64 = []
            if input_tensors:
                input_images_b64 = self.image_codec.prepare_input_images(input_tensors)

            # 自动检测宽高比
            effective_aspect_ratio = aspect_ratio
            if aspect_ratio == "Auto" and input_tensors:
                first_img = input_tensors[0]
                if first_img is not None and len(first_img.shape) >= 3:
                    h, w = first_img.shape[1], first_img.shape[2]
                    ratio = w / h
                    aspect_ratios = {
                        "1:1": 1.0, "3:2": 1.5, "2:3": 0.667, "3:4": 0.75,
                        "4:3": 1.333, "4:5": 0.8, "5:4": 1.25, "9:16": 0.5625,
                        "16:9": 1.778, "21:9": 2.333
                    }
                    closest = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
                    effective_aspect_ratio = closest[0]
                    logger.info(f"自动检测宽高比: {effective_aspect_ratio} (原始比例: {ratio:.2f})")

            # 获取最大工作线程数
            max_workers = self.config_manager.load_max_workers()

            # 根据协议选择客户端
            if api_protocol == "Gemini":
                api_client = GeminiApiClient(
                    self.config_manager,
                    logger,
                    self.interrupt_checker
                )
            else:  # OpenAI
                api_client = OpenAIApiClient(
                    self.config_manager,
                    logger,
                    self.interrupt_checker
                )

            # 创建批量生成运行器
            batch_runner = BatchGenerationRunner(
                api_client,
                max_workers=max_workers,
                logger_instance=logger,
                interrupt_checker=self.interrupt_checker
            )

            # 处理超时
            timeout_value = None if 超时秒数 == 0 else (15, 超时秒数)

            logger.header("🎨 高级图像生成任务")
            logger.info(f"协议: {api_protocol}")
            logger.info(f"模型: {model_type}")
            logger.info(f"批次大小: {batch_size}")
            logger.info(f"宽高比: {effective_aspect_ratio}")
            logger.info(f"分辨率: {imageSize}")
            if input_tensors:
                logger.info(f"参考图像: {len(input_tensors)} 张")
            if seed >= 0:
                logger.info(f"种子: {seed}")
            logger.separator()

            # 创建请求数据模板
            if api_protocol == "Gemini":
                # Gemini 协议需要同时传递 size（宽高比）和 quality（质量映射到分辨率）
                # 修复：使用正确的参数名 size 和 quality
                request_data_template = api_client.create_request_data(
                    model=model_type,
                    prompt=prompt,
                    size=effective_aspect_ratio if effective_aspect_ratio != "Auto" else None,
                    quality=imageSize,  # "1K"/"2K"/"4K" 会自动映射
                    input_images_b64=input_images_b64 if input_images_b64 else None,
                    top_p=top_p if top_p != 1.0 else None,
                    seed=seed if seed >= 0 else None,
                )
                
                # 关键修复：确保 size 和 image_size 参数被正确传递
                if effective_aspect_ratio != "Auto" and "size" not in request_data_template:
                    request_data_template["size"] = effective_aspect_ratio
                if imageSize != "无" and "image_size" not in request_data_template:
                    # 映射 imageSize 到 image_size
                    request_data_template["image_size"] = imageSize
            else:  # OpenAI
                # OpenAI 协议需要转换宽高比为尺寸
                size_map = {
                    "1:1": "1024x1024",
                    "16:9": "1024x576",
                    "9:16": "576x1024",
                    "2:3": "768x1152",
                    "3:2": "1152x768",
                    "4:3": "1024x768",
                    "3:4": "768x1024",
                    "4:5": "832x1040",
                    "5:4": "1040x832",
                    "21:9": "1344x576",
                    "Auto": "1024x1024",
                }

                size = size_map.get(effective_aspect_ratio, "1024x1024")

                request_data_template = api_client.create_request_data(
                    model=model_type,
                    prompt=prompt,
                    size=size,
                    input_images_b64=input_images_b64 if input_images_b64 else None,
                    top_p=top_p if top_p != 1.0 else None,
                    seed=seed if seed >= 0 else None,
                )

            # 执行批量生成
            batch_results, batch_errors = batch_runner.run_batch(
                api_key=resolved_api_key,
                request_data_template=request_data_template,
                api_base_url=resolved_api_base_url,
                batch_size=batch_size,
                timeout=timeout_value,
                bypass_proxy=绕过代理,
                verify_ssl=True,
            )

            self._ensure_not_interrupted()

            # 提取所有图片
            all_images = []
            for result in batch_results:
                images = self._extract_images_from_response(result["response"], api_protocol)
                all_images.extend(images)

            if not all_images:
                error_msg = f"生成失败：未返回任何图片"
                if batch_errors:
                    error_msg += f"\n错误详情：\n" + "\n".join(batch_errors[:3])

                error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
                return (error_tensor, error_msg)

            # 解码图片
            self._ensure_not_interrupted()
            image_tensor = self.image_codec.base64_to_tensor_parallel(
                all_images,
                log_prefix="高级生成",
                max_workers=max_workers
            )

            total_time = time.time() - start_time
            actual_count = len(all_images)
            avg_time = total_time / actual_count if actual_count > 0 else 0

            # 构建返回信息
            info_text = f"✅ 成功生成 {actual_count} 张图像\n"
            info_text += f"协议: {api_protocol}\n"
            info_text += f"模型: {model_type}\n"
            info_text += f"宽高比: {effective_aspect_ratio}\n"
            if imageSize != "无":
                info_text += f"分辨率: {imageSize}\n"
            info_text += f"批次大小: {batch_size}\n"
            info_text += f"总耗时: {total_time:.2f}s，平均 {avg_time:.2f}s/张"

            if batch_errors:
                info_text += f"\n\n⚠️ 部分请求失败 ({len(batch_errors)} 个)"

            # 显示完成统计
            logger.summary("任务完成", {
                "成功生成": f"{actual_count} 张",
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
            error_tensor = self.error_canvas.build_error_tensor_from_text("生成失败", error_msg)
            return (error_tensor, error_msg)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFOLIE_AdvancedImageGeneration": ImageGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFOLIE_AdvancedImageGeneration": "🎨 AFOLIE 高级图像生成",
}
