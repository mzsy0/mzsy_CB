"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

# 用于通过额外的py定义的文件，获取外部数据和加载的规则
import importlib


def get_ext_data_by_name(name) -> dict:
    try:
        # 构造模块名
        module_name = f"外部数据.{name}"

        # 动态导入模块
        ext_data_module = importlib.import_module(module_name)

        return {name: (getattr(ext_data_module, "read_ext_data"), getattr(ext_data_module, "ext_data_path"))}
    except ModuleNotFoundError:
        return {}
    except AttributeError:
        raise ValueError(f"Error accessing external data content in module {name}.")


def load_ext_data() -> dict:
    """
    加载所有外部自定义数据，遍历"外部数据"文件夹，并且整合成dict
    """
    from pathlib import Path

    ext_data_content = {}
    ext_data_path = Path("外部数据")

    # 检查文件夹是否存在
    if not ext_data_path.exists():
        return ext_data_content

    # 遍历文件夹，只处理.py文件且不以_开头
    for file_path in ext_data_path.glob("*.py"):
        if not file_path.name.startswith("_"):
            # 获取文件名（不含扩展名）
            file_name = file_path.stem
            ext_data_content.update(get_ext_data_by_name(file_name))

    return ext_data_content
