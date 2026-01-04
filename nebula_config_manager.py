"""
配置管理器
管理 API Key 和其他配置
"""

import configparser
import os
import sys
import re
from typing import Optional

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from nebula_logger import logger


class ConfigManager:
    """集中管理 Nebula API 节点的配置"""

    _CONFIG_SECTION = "nebula"
    _DEFAULT_API_KEY = "your-api-key-here"
    _DEFAULT_API_BASE_URL = "https://llm.ai-nebula.com/v1"
    _PLACEHOLDER_KEYS = {
        "your-api-key-here",
        "your_api_key_here",
        "yourapikeyhere",
        "sk-xxxxxxxxxx",
    }

    def __init__(self, base_dir: Optional[str] = None):
        self._base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self._config_path = os.path.join(self._base_dir, "config.ini")

    def _ensure_sample_config_exists(self) -> None:
        """确保配置文件存在"""
        if os.path.exists(self._config_path):
            return
        config = configparser.ConfigParser()
        cpu_limit = max(1, os.cpu_count() or 4)
        default_workers = min(8, cpu_limit)
        config[self._CONFIG_SECTION] = {
            "api_key": self._DEFAULT_API_KEY,
            "api_base_url": self._DEFAULT_API_BASE_URL,
            "max_workers": str(default_workers),
        }
        try:
            with open(self._config_path, "w", encoding="utf-8") as handle:
                config.write(handle)
            logger.success(f"已创建示例配置文件: {self._config_path}")
            logger.info("请编辑文件并填入你的 API Key")
        except Exception as exc:
            logger.warning(f"创建配置文件失败: {exc}")

    def sanitize_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """清理和验证 API Key"""
        if not api_key:
            return None
        cleaned = api_key.strip()
        if not cleaned:
            return None
        normalized = cleaned.lower()
        compact = re.sub(r"[\s_-]+", "", normalized)
        if normalized in self._PLACEHOLDER_KEYS or compact in self._PLACEHOLDER_KEYS:
            return None
        return cleaned

    def load_api_key(self) -> str:
        """加载 API Key"""
        self._ensure_sample_config_exists()
        parser = configparser.ConfigParser()
        if os.path.exists(self._config_path):
            try:
                parser.read(self._config_path, encoding="utf-8")
                if parser.has_section(self._CONFIG_SECTION):
                    return parser.get(
                        self._CONFIG_SECTION,
                        "api_key",
                        fallback=self._DEFAULT_API_KEY
                    )
            except Exception as exc:
                logger.warning(f"读取配置文件失败: {exc}")
        return self._DEFAULT_API_KEY

    def get_effective_api_base_url(self) -> str:
        """获取有效的 API Base URL"""
        parser = configparser.ConfigParser()
        if os.path.exists(self._config_path):
            try:
                parser.read(self._config_path, encoding="utf-8")
                if parser.has_section(self._CONFIG_SECTION):
                    url = parser.get(
                        self._CONFIG_SECTION,
                        "api_base_url",
                        fallback=""
                    ).strip()
                    if url:
                        return url
            except Exception as exc:
                logger.warning(f"读取 config 中的 api_base_url 失败: {exc}")
        return self._DEFAULT_API_BASE_URL

    def load_max_workers(self) -> int:
        """加载最大工作线程数"""
        cpu_limit = max(1, os.cpu_count() or 1)
        default_workers = min(8, cpu_limit)
        parser = configparser.ConfigParser()
        if os.path.exists(self._config_path):
            try:
                parser.read(self._config_path, encoding="utf-8")
                if parser.has_section(self._CONFIG_SECTION):
                    value = parser.getint(
                        self._CONFIG_SECTION,
                        "max_workers",
                        fallback=default_workers
                    )
                    return max(1, min(value, cpu_limit))
            except Exception as exc:
                logger.warning(f"读取 config 中的 max_workers 失败: {exc}")
        return default_workers
