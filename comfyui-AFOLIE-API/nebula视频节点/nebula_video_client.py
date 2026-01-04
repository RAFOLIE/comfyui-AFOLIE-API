"""
Nebula 视频生成 API 客户端
支持多种模型：Sora 2, Veo, 阿里万相, 豆包 Seedance
"""

from __future__ import annotations

import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

import requests

import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(MODULE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from nebula_logger import logger
except ImportError:
    logger = None


class VideoModel(Enum):
    """支持的视频生成模型"""
    SORA_2 = "sora-2"
    VEO_3_0_FAST = "veo-3.0-fast-generate-001"
    VEO_3_1_FAST = "veo-3.1-fast-generate-preview"
    WAN_T2V = "wan2.5-t2v-preview"
    WAN_I2V = "wan2.5-i2v-preview"
    DOUBAO_LITE_T2V = "doubao-seedance-1-0-lite-t2v-250428"
    DOUBAO_LITE_I2V = "doubao-seedance-1-0-lite-i2v-250428"
    DOUBAO_PRO = "doubao-seedance-1-0-pro-250528"


class VideoClient:
    """Nebula 视频生成 API 客户端"""

    def __init__(
        self,
        api_base_url: str = "https://llm.ai-nebula.com",
        api_key: str = "",
        interrupt_checker=None
    ):
        # 确保 URL 以 / 结尾，但不要包含 /v1
        self.api_base_url = api_base_url.rstrip('/').rstrip('/v1')
        self.api_key = api_key
        self.interrupt_checker = interrupt_checker

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _ensure_not_interrupted(self):
        """检查是否被中断"""
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    def submit_video_task(
        self,
        model: str,
        prompt: Optional[str] = None,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        image: Optional[str] = None,
        last_frame: Optional[str] = None,
        seed: Optional[int] = None,
        fps: Optional[int] = None,
        generate_audio: Optional[bool] = None,
        add_watermark: Optional[bool] = None,
        person_generation: Optional[str] = None,
        sample_count: Optional[int] = None,
        size: Optional[str] = None,
        seconds: Optional[str] = None,
        input_reference: Optional[str] = None,
        remix_from_video_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        smart_rewrite: Optional[bool] = None,
        generate_audio_wan: Optional[bool] = None,
        audio_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """提交视频生成任务"""
        self._ensure_not_interrupted()

        url = f"{self.api_base_url}/v1/video/generations"

        headers = self._build_headers()

        # 构建请求体
        request_body = {"model": model}

        # 通用参数
        if prompt is not None:
            request_body["prompt"] = prompt
        if image is not None:
            request_body["image"] = image
        if seed is not None and seed >= 0:
            request_body["seed"] = seed

        # Sora 2 专用参数
        if model == VideoModel.SORA_2.value:
            if seconds is not None:
                request_body["seconds"] = seconds
            if size is not None:
                request_body["size"] = size
            if input_reference is not None:
                request_body["input_reference"] = input_reference
            if remix_from_video_id is not None:
                request_body["remix_from_video_id"] = remix_from_video_id

        # Veo 专用参数
        elif model.startswith("veo"):
            if duration_seconds is not None:
                request_body["durationSeconds"] = duration_seconds
            if aspect_ratio is not None:
                request_body["aspectRatio"] = aspect_ratio
            if resolution is not None:
                request_body["resolution"] = resolution
            if fps is not None:
                request_body["fps"] = fps
            if generate_audio is not None:
                request_body["generateAudio"] = generate_audio
            if person_generation is not None:
                request_body["personGeneration"] = person_generation
            if add_watermark is not None:
                request_body["addWatermark"] = add_watermark
            if sample_count is not None:
                request_body["sampleCount"] = sample_count
            if last_frame is not None:
                request_body["lastFrame"] = last_frame
            if image is not None:
                request_body["image"] = image

        # 阿里万相专用参数
        elif model.startswith("wan"):
            if duration is not None:
                request_body["duration"] = duration
            if resolution is not None:
                request_body["resolution"] = resolution
            if smart_rewrite is not None:
                request_body["smart_rewrite"] = smart_rewrite
            if generate_audio_wan is not None:
                request_body["generate_audio"] = generate_audio_wan
            if audio_url is not None:
                request_body["audio_url"] = audio_url
            if size is not None:
                request_body["size"] = size

        # 豆包 Seedance 专用参数
        elif model.startswith("doubao"):
            if metadata is not None:
                request_body["metadata"] = metadata
            else:
                # 兼容模式：使用 prompt 并转换为 metadata
                if prompt:
                    request_body["metadata"] = self._create_doubao_metadata(prompt)

        # 发送请求
        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")

        if timeout is None:
            timeout = (15, 60)

        try:
            self._ensure_not_interrupted()

            # 使用线程发送请求以支持中断
            done_event = threading.Event()
            resp_holder = {}
            exc_holder = {}

            def _do_request():
                try:
                    resp_holder["resp"] = requests.post(
                        url,
                        data=payload,
                        headers=headers,
                        timeout=timeout,
                        verify=True,
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
                try:
                    error_json = e.response.json()
                    if 'error' in error_json:
                        error_msg += f"\n错误详情: {error_json['error']}"
                except:
                    pass
            except Exception:
                pass
            raise RuntimeError(error_msg)

        except requests.RequestException as e:
            error_msg = f"请求失败: {type(e).__name__} - {str(e)}"
            raise RuntimeError(error_msg)

    def _create_doubao_metadata(self, prompt: str) -> Dict[str, Any]:
        """创建豆包 Seedance 的 metadata 格式"""
        return {
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }

    def query_video_task(
        self,
        task_id: str,
        timeout: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """查询视频任务状态"""
        self._ensure_not_interrupted()

        url = f"{self.api_base_url}/v1/video/generations/{task_id}"

        headers = self._build_headers()

        if timeout is None:
            timeout = (15, 30)

        try:
            self._ensure_not_interrupted()

            done_event = threading.Event()
            resp_holder = {}
            exc_holder = {}

            def _do_request():
                try:
                    resp_holder["resp"] = requests.get(
                        url,
                        headers=headers,
                        timeout=timeout,
                        verify=True,
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
            error_msg = f"查询失败: {e.response.status_code}"
            try:
                error_text = e.response.text[:500]
                error_msg += f"\n响应内容: {error_text}"
            except Exception:
                pass
            raise RuntimeError(error_msg)

        except requests.RequestException as e:
            error_msg = f"查询失败: {type(e).__name__} - {str(e)}"
            raise RuntimeError(error_msg)

    def wait_for_task_completion(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        max_wait_time: float = 3600.0,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """等待任务完成"""
        self._ensure_not_interrupted()

        start_time = time.time()
        last_log_time = start_time

        while True:
            self._ensure_not_interrupted()

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time:
                raise RuntimeError(f"任务超时（已等待 {elapsed:.0f} 秒）")

            # 查询任务状态
            result = self.query_video_task(task_id)

            # 调用进度回调
            if progress_callback:
                progress_callback(result, elapsed)

            # 记录日志（每30秒记录一次）
            current_time = time.time()
            if current_time - last_log_time >= 30:
                status = result.get("status", "unknown")
                if logger:
                    logger.info(f"任务状态: {status} (已等待 {elapsed:.0f}s)")
                last_log_time = current_time

            # 检查是否完成
            status = result.get("status", "").lower()

            # 成功状态
            if status == "succeeded" or status == "completed":
                if logger:
                    logger.info(f"✅ 任务完成！总耗时: {elapsed:.0f}s")
                return result

            # 失败状态
            elif status == "failed":
                error_msg = result.get("error", "任务失败")
                raise RuntimeError(f"任务失败: {error_msg}")

            # 继续等待
            time.sleep(poll_interval)
