"""
ComfyUI AFOLIE API 节点加载器
支持 Nebula 图片生成 API
"""

import os
import sys
import importlib.util
from pathlib import Path

# 导入日志系统
from .nebula_logger import logger

# 获取当前文件夹路径
current_dir = Path(__file__).parent

# 确保当前目录在 sys.path 中
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 初始化节点映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__version__ = "1.0.0"

# 需要跳过的文件列表
SKIP_FILES = {
    "__init__.py",
    "nebula_logger.py",
    "nebula_config_manager.py",
    "nebula_api_client.py",
    "nebula_image_codec.py",
}

# 需要加载的子目录列表
LOAD_SUBDIRS = {"nebula视频节点"}

# 显示加载器标题
logger.header("🌌 AFOLIE Nebula API 节点加载器")
logger.info(f"版本 {__version__}")

# 自动查找并加载所有Python文件中的节点
for py_file in current_dir.glob("*.py"):
    # 跳过特殊文件
    if py_file.name in SKIP_FILES:
        continue

    try:
        # 动态导入模块
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 合并节点映射
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        logger.success(f"成功加载节点文件: {py_file.name}")

    except Exception as e:
        logger.error(f"加载节点文件失败 {py_file.name}: {str(e)}")

# 加载子目录中的节点
for subdir_name in LOAD_SUBDIRS:
    subdir_path = current_dir / subdir_name
    if subdir_path.exists() and subdir_path.is_dir():
        logger.info(f"正在扫描子目录: {subdir_name}")
        
        # 尝试导入子目录的 __init__.py
        init_file = subdir_path / "__init__.py"
        if init_file.exists():
            try:
                # 动态导入子目录模块
                module_name = f"comfyui_AFOLIE_API.{subdir_name}"
                spec = importlib.util.spec_from_file_location(module_name, init_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # 合并节点映射
                if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

                if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

                logger.success(f"成功加载子目录节点: {subdir_name}")

            except Exception as e:
                logger.error(f"加载子目录节点失败 {subdir_name}: {str(e)}")
        else:
            logger.warning(f"子目录 {subdir_name} 缺少 __init__.py 文件")

# 打印加载的节点信息
if NODE_CLASS_MAPPINGS:
    logger.info(f"总共加载了 {len(NODE_CLASS_MAPPINGS)} 个自定义节点")
    for node_name in NODE_CLASS_MAPPINGS.keys():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
        logger.info(f"   - {display_name} ({node_name})")
else:
    logger.warning("未找到任何有效的节点")

# ComfyUI需要的变量
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']
WEB_DIRECTORY = "./web"
