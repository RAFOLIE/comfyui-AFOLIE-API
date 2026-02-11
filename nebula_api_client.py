"""
Nebula API 客户端
封装与 Nebula 图像生成接口的 HTTP 交互
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.exceptions import InsecureRequestWarning

try:
    from requests.packages import urllib3 as requests_urllib3
except Exception:
    requests_urllib3 = None

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from nebula_logger import logger


class NebulaApiClient:
    """封装与 Nebula 图像生成接口交互的 HTTP 客户端"""

    _DEFAULT_CONNECT_TIMEOUT = 15.0
    _DEFAULT_READ_TIMEOUT = 420.0
    _MAX_RETRIES = 2
    _BASE_BACKOFF = 2.0
    _RETRYABLE_STATUS = {429, 500}
    _INSECURE_WARNING_SUPPRESSED = False
    
    # HTTP 状态码说明
    _STATUS_CODE_MESSAGES = {
        200: "请求成功",
        401: "API Key 无效或已过期",
        429: "请求频率过高",
        500: "服务器内部错误",
    }

    def __init__(self, config_manager, logger_instance=logger, interrupt_checker=None) -> None:
        self.config_manager = config_manager
        self.logger = logger_instance
        self.interrupt_checker = interrupt_checker
        self._thread_local = threading.local()

    def _get_session(self, bypass_proxy: bool = False) -> requests.Session:
        attr_name = "session_no_proxy" if bypass_proxy else "session"
        session = getattr(self._thread_local, attr_name, None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            if bypass_proxy:
                session.trust_env = False
                session.proxies = {}
            setattr(self._thread_local, attr_name, session)
        return session

    def _ensure_not_interrupted(self) -> None:
        if self.interrupt_checker is not None:
            self.interrupt_checker()

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _resolve_timeout(self, timeout: Optional[Any]) -> Tuple[float, Optional[float]]:
        if isinstance(timeout, (tuple, list)) and len(timeout) == 2:
            connect = float(timeout[0]) if timeout[0] else self._DEFAULT_CONNECT_TIMEOUT
            if timeout[1] is None:
                read = None
            else:
                read = float(timeout[1]) if timeout[1] else self._DEFAULT_READ_TIMEOUT
        elif isinstance(timeout, (int, float)) and timeout > 0:
            connect = read = float(timeout)
        else:
            connect = self._DEFAULT_CONNECT_TIMEOUT
            read = self._DEFAULT_READ_TIMEOUT
        connect = max(1.0, connect)
        if read is not None:
            read = max(5.0, read)
        return (connect, read)

    @classmethod
    def _suppress_insecure_warning(cls, verify_ssl: bool) -> None:
        if verify_ssl or cls._INSECURE_WARNING_SUPPRESSED:
            return
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        urllib3.disable_warnings(InsecureRequestWarning)
        if requests_urllib3 is not None:
            try:
                requests_urllib3.disable_warnings(InsecureRequestWarning)
            except Exception:
                pass
        cls._INSECURE_WARNING_SUPPRESSED = True

    def _interruptible_post(
        self,
        session: requests.Session,
        url: str,
        payload: bytes,
        headers: Dict[str, str],
        timeout: Tuple[float, Optional[float]],
        verify: bool,
        bypass_proxy: bool,
    ) -> requests.Response:
        if self.interrupt_checker is None:
            return session.post(
                url,
                data=payload,
                headers=headers,
                timeout=timeout,
                verify=verify,
            )

        done_event = threading.Event()
        resp_holder: Dict[str, Any] = {}
        exc_holder: Dict[str, BaseException] = {}

        def _do_request() -> None:
            try:
                resp_holder["resp"] = session.post(
                    url,
                    data=payload,
                    headers=headers,
                    timeout=timeout,
                    verify=verify,
                )
            except BaseException as exc:
                exc_holder["exc"] = exc
            finally:
                done_event.set()

        thread = threading.Thread(target=_do_request, daemon=True)
        thread.start()

        poll_interval = 0.25
        attr_name = "session_no_proxy" if bypass_proxy else "session"
        try:
            while not done_event.wait(timeout=poll_interval):
                self._ensure_not_interrupted()
            self._ensure_not_interrupted()
        except BaseException:
            try:
                session.close()
            finally:
                setattr(self._thread_local, attr_name, None)
            raise

        if "exc" in exc_holder:
            raise exc_holder["exc"]

        resp = resp_holder.get("resp")
        if resp is None:
            raise RuntimeError("请求被中断或未获得响应")
        return resp

    def create_request_data(
        self,
        model: str,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        response_format: str = "b64_json",
        input_images_b64: Optional[List[str]] = None,
        **extra_params
    ) -> Dict[str, Any]:
        """创建请求数据"""
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

        # 处理图生图的 contents 参数
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

        # 添加额外参数
        for key, value in extra_params.items():
            if value is not None:
                request_body[key] = value

        return request_body

    def _extract_error_text(self, response: requests.Response, max_length: int = 500) -> str:
        """从响应中提取错误信息"""
        try:
            if not response:
                return "无响应"
            
            # 尝试解析 JSON 响应
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" in content_type:
                try:
                    json_data = response.json()
                    # 尝试常见的错误字段
                    for field in ["error", "message", "detail", "error_description", "error_msg"]:
                        if field in json_data:
                            error_value = json_data[field]
                            if isinstance(error_value, str):
                                return error_value[:max_length]
                            elif isinstance(error_value, dict):
                                # 嵌套的错误对象
                                if "message" in error_value:
                                    return error_value["message"][:max_length]
                    # 如果都没有，返回整个 JSON 的字符串表示
                    return str(json_data)[:max_length]
                except Exception:
                    pass
            
            # 处理 HTML 响应（比如 500 错误返回的错误页面）
            if "text/html" in content_type:
                text = response.text
                # 尝试提取 <title> 标签内容
                title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
                    # 清理 HTML 实体和多余空白
                    title = re.sub(r'\s+', ' ', title)
                    return title[:max_length]
                # 尝试提取 <h1> 标签内容
                h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', text, re.IGNORECASE | re.DOTALL)
                if h1_match:
                    h1 = h1_match.group(1).strip()
                    h1 = re.sub(r'\s+', ' ', h1)
                    return h1[:max_length]
                # 返回 HTML 的前 200 个字符
                return text[:200] + "..."
            
            # 如果不是 JSON 或 HTML，返回文本
            text = response.text.strip()
            if text:
                return text[:max_length]
            
            # 如果都没有，返回状态码
            return f"HTTP {response.status_code}"
        except Exception:
            return f"HTTP {response.status_code}"

    def send_request(
        self,
        api_key: str,
        request_data: Dict[str, Any],
        api_base_url: str,
        timeout: Optional[Any] = None,
        bypass_proxy: bool = False,
        verify_ssl: bool = True,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """发送 API 请求"""
        sanitized_key = self.config_manager.sanitize_api_key(api_key)
        if not sanitized_key:
            raise ValueError("请填写有效的 API Key")

        url = f"{api_base_url.rstrip('/')}/images/generations"
        session = self._get_session(bypass_proxy)
        self._suppress_insecure_warning(verify_ssl)
        connect_timeout, read_timeout_global = self._resolve_timeout(timeout)
        headers = self._build_headers(sanitized_key)

        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")
        last_error: Optional[BaseException] = None

        effective_max_retries = (
            max_retries
            if isinstance(max_retries, int) and max_retries >= 1
            else self._MAX_RETRIES
        )

        global_start = time.time()
        attempt_delay = self._BASE_BACKOFF

        for attempt in range(1, effective_max_retries + 1):
            self._ensure_not_interrupted()
            elapsed = time.time() - global_start
            if read_timeout_global is None:
                remaining_read = None
            else:
                remaining_read = read_timeout_global - elapsed
                if remaining_read <= 0:
                    raise RuntimeError(
                        f"请求超时：总耗时 {elapsed:.1f}s 已超过读取上限 {read_timeout_global:.1f}s"
                    )

            try:
                response = self._interruptible_post(
                    session,
                    url,
                    payload,
                    headers,
                    (connect_timeout, remaining_read),
                    verify_ssl,
                    bypass_proxy,
                )
                # 检查响应状态码
                status_code = response.status_code
                
                # 200 成功
                if status_code == 200:
                    return response.json()
                
                # 401 API Key 错误 - 不重试
                elif status_code == 401:
                    error_text = self._extract_error_text(response)
                    raise RuntimeError(f"API Key 无效或已过期\n服务器返回：{error_text}")
                
                # 429 请求频率过高 - 重试
                elif status_code == 429:
                    if attempt < effective_max_retries:
                        # 429 错误使用更长的等待时间
                        retry_wait = min(attempt_delay * 2, 30)  # 最多等待30秒
                        self.logger.warning(f"请求频率过高，{retry_wait:.1f}秒后重试 ({attempt}/{effective_max_retries})")
                        time.sleep(retry_wait)
                        attempt_delay *= 2  # 下次等待时间翻倍
                    else:
                        error_text = self._extract_error_text(response)
                        raise RuntimeError(f"请求频率过高，已达到最大重试次数\n服务器返回：{error_text}")
                
                # 500 服务器内部错误 - 重试
                elif status_code == 500:
                    if attempt < effective_max_retries:
                        error_text = self._extract_error_text(response)
                        self.logger.warning(f"服务器内部错误（{error_text}），将重试 ({attempt}/{effective_max_retries})")
                        time.sleep(attempt_delay)
                        attempt_delay *= 1.5
                    else:
                        error_text = self._extract_error_text(response)
                        raise RuntimeError(f"服务器内部错误，已达到最大重试次数\n服务器返回：{error_text}")
                
                # 其他错误状态码 - 不重试
                else:
                    raise RuntimeError(f"HTTP {status_code}\n服务器返回：{self._extract_error_text(response)}")
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt < effective_max_retries:
                    self.logger.warning(f"请求失败，将重试 ({attempt}/{effective_max_retries})")
                    time.sleep(attempt_delay)
                    attempt_delay *= 1.5
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response else None
                if status in self._RETRYABLE_STATUS and attempt < effective_max_retries:
                    if status == 429:
                        retry_wait = min(attempt_delay * 2, 30)
                        self.logger.warning(f"HTTP {status}，{retry_wait:.1f}秒后重试")
                        time.sleep(retry_wait)
                        attempt_delay *= 2
                    elif status == 500:
                        self.logger.warning(f"HTTP {status}，将重试")
                        time.sleep(attempt_delay)
                        attempt_delay *= 1.5
                else:
                    error_text = self._extract_error_text(exc.response) if exc.response else ""
                    raise RuntimeError(f"HTTP {status}\n服务器返回：{error_text}")
            except requests.RequestException as exc:
                last_error = exc
                raise RuntimeError(f"HTTP 请求失败，请检查网络连接")

        error_label = type(last_error).__name__ if last_error is not None else "未知错误"
        raise RuntimeError(f"连续 {effective_max_retries} 次请求失败（{error_label}）")

    def extract_images(self, response_data: Dict[str, Any]) -> Tuple[List[str], str]:
        """从 API 响应中提取图片"""
        images: List[str] = []
        
        # 处理标准响应格式
        if "data" in response_data:
            data = response_data["data"]
            if isinstance(data, dict) and "data" in data:
                # 嵌套的 data 结构
                data = data["data"]
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # b64_json 格式
                        if "b64_json" in item and item["b64_json"]:
                            images.append(item["b64_json"])
                        # url 格式 - 需要下载
                        elif "url" in item and item["url"]:
                            b64_data = self._download_image_to_base64(item["url"])
                            if b64_data:
                                images.append(b64_data)

        # 获取修订后的提示词
        revised_prompt = ""
        if "data" in response_data:
            data = response_data["data"]
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    revised_prompt = data[0].get("revised_prompt", "")

        return images, revised_prompt

    def _download_image_to_base64(self, url: str, timeout: float = 30.0) -> Optional[str]:
        """下载图片并转换为 base64"""
        try:
            session = self._get_session(bypass_proxy=False)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "image/*,*/*;q=0.8",
            }
            response = session.get(url, headers=headers, timeout=timeout, verify=True)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                content = response.content
                if len(content) < 8:
                    return None
                if not (content[:4] == b'\x89PNG' or 
                        content[:3] == b'\xff\xd8\xff' or 
                        content[:4] == b'GIF8' or
                        (content[:4] == b'RIFF' and content[8:12] == b'WEBP')):
                    return None
            
            image_data = response.content
            base64_data = base64.b64encode(image_data).decode('utf-8')
            self.logger.info(f"图片下载成功：{len(image_data)} 字节")
            return base64_data
        except Exception as exc:
            self.logger.warning(f"图片下载失败：{type(exc).__name__}")
            return None


__all__ = ["NebulaApiClient"]
