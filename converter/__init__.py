"""
Torch2Tensorflow - PyTorch模型转TensorFlow模型转换器
支持多种环境配置 (Mac/Linux, TensorFlow 2.1.x/2.13.x)
"""

import os
import sys

# 检测TensorFlow版本并选择合适的转换器
def _get_converter_class():
    """根据环境选择合适的转换器类"""
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        
        # 检查是否存在TensorFlow 2.1.3兼容版本
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'torch_to_tf_v21.py')):
            if tf_version.startswith('2.1'):
                # 使用TensorFlow 2.1.3兼容版本
                from .torch_to_tf_v21 import TorchToTensorFlowConverter
                print(f"使用TensorFlow 2.1.3兼容转换器 (TF版本: {tf_version})")
                return TorchToTensorFlowConverter
        
        # 使用通用版本
        from .torch_to_tf import TorchToTensorFlowConverter
        print(f"使用通用转换器 (TF版本: {tf_version})")
        return TorchToTensorFlowConverter
        
    except ImportError:
        # 如果TensorFlow未安装，使用通用版本
        from .torch_to_tf import TorchToTensorFlowConverter
        print("TensorFlow未安装，使用通用转换器")
        return TorchToTensorFlowConverter

# 导入转换器类
TorchToTensorFlowConverter = _get_converter_class()

# 导入工具类
from .model_utils import ModelUtils

__version__ = "1.0.0"
__author__ = "PyTorch2TensorFlow Team"

# 环境信息
def get_environment_info():
    """获取当前环境信息"""
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform
    }
    
    try:
        import torch
        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "未安装"
    
    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
    except ImportError:
        info["tensorflow_version"] = "未安装"
    
    try:
        import onnx
        info["onnx_version"] = onnx.__version__
    except ImportError:
        info["onnx_version"] = "未安装"
    
    return info

__all__ = [
    "TorchToTensorFlowConverter",
    "ModelUtils",
    "get_environment_info"
] 