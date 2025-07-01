"""
PyTorch到TensorFlow模型转换器
"""

import torch
import tensorflow as tf
import onnx
import tf2onnx
import tempfile
import os
from typing import Optional, Tuple, Dict, Any
import numpy as np


class TorchToTensorFlowConverter:
    """
    PyTorch模型到TensorFlow模型的转换器
    """
    
    def __init__(self):
        """初始化转换器"""
        self.torch_model = None
        self.tf_model = None
        self.input_shape = None
        
    def load_pytorch_model(self, model_path: str, model_class=None, **kwargs):
        """
        加载PyTorch模型
        
        Args:
            model_path: PyTorch模型文件路径
            model_class: 模型类（如果需要从头构建）
            **kwargs: 模型构建参数
        """
        try:
            if model_class is not None:
                # 从模型类构建
                self.torch_model = model_class(**kwargs)
                state_dict = torch.load(model_path, map_location='cpu')
                self.torch_model.load_state_dict(state_dict)
            else:
                # 直接加载完整模型
                self.torch_model = torch.load(model_path, map_location='cpu')
            
            self.torch_model.eval()
            print(f"成功加载PyTorch模型: {model_path}")
            return True
            
        except Exception as e:
            print(f"加载PyTorch模型失败: {e}")
            return False
    
    def set_input_shape(self, input_shape: Tuple[int, ...]):
        """
        设置输入形状
        
        Args:
            input_shape: 输入张量形状，例如 (1, 3, 224, 224)
        """
        self.input_shape = input_shape
        print(f"设置输入形状: {input_shape}")
    
    def convert_via_onnx(self, output_path: str, input_names: list = None, 
                        output_names: list = None, opset_version: int = 11) -> bool:
        """
        通过ONNX中间格式转换模型
        
        Args:
            output_path: TensorFlow模型输出路径
            input_names: 输入节点名称列表
            output_names: 输出节点名称列表
            opset_version: ONNX opset版本
            
        Returns:
            bool: 转换是否成功
        """
        if self.torch_model is None:
            print("错误：请先加载PyTorch模型")
            return False
            
        if self.input_shape is None:
            print("错误：请先设置输入形状")
            return False
        
        try:
            # 创建临时ONNX文件
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_onnx:
                onnx_path = tmp_onnx.name
            
            # Step 1: PyTorch -> ONNX
            print("步骤1: 将PyTorch模型转换为ONNX...")
            dummy_input = torch.randn(self.input_shape)
            
            torch.onnx.export(
                self.torch_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names or ['input'],
                output_names=output_names or ['output']
            )
            
            # 验证ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证成功")
            
            # Step 2: ONNX -> TensorFlow
            print("步骤2: 将ONNX模型转换为TensorFlow...")
            
            # 使用tf2onnx进行转换
            from tf2onnx import convert
            
            model_proto, _ = tf2onnx.convert.from_onnx(onnx_path)
            
            # 保存TensorFlow模型
            tf.saved_model.save(model_proto, output_path)
            
            print(f"成功转换模型到: {output_path}")
            
            # 清理临时文件
            os.unlink(onnx_path)
            
            return True
            
        except Exception as e:
            print(f"转换失败: {e}")
            return False
    
    def convert_direct(self, output_path: str) -> bool:
        """
        直接转换方法（实验性功能）
        
        Args:
            output_path: TensorFlow模型输出路径
            
        Returns:
            bool: 转换是否成功
        """
        print("直接转换方法目前为实验性功能")
        # 这里可以添加直接转换的逻辑
        # 例如使用torch.jit.trace然后手动构建TensorFlow图
        return False
    
    def validate_conversion(self, tf_model_path: str, test_inputs: np.ndarray = None) -> bool:
        """
        验证转换结果
        
        Args:
            tf_model_path: TensorFlow模型路径
            test_inputs: 测试输入数据
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 加载TensorFlow模型
            tf_model = tf.saved_model.load(tf_model_path)
            
            # 生成测试输入
            if test_inputs is None:
                test_inputs = np.random.randn(*self.input_shape).astype(np.float32)
            
            # PyTorch推理
            with torch.no_grad():
                torch_input = torch.from_numpy(test_inputs)
                torch_output = self.torch_model(torch_input).numpy()
            
            # TensorFlow推理
            tf_input = tf.constant(test_inputs)
            tf_output = tf_model(tf_input).numpy()
            
            # 比较输出
            diff = np.abs(torch_output - tf_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"输出差异统计:")
            print(f"  最大差异: {max_diff:.6f}")
            print(f"  平均差异: {mean_diff:.6f}")
            
            # 设置容差阈值
            tolerance = 1e-4
            if max_diff < tolerance:
                print("✓ 验证通过：模型转换成功")
                return True
            else:
                print("✗ 验证失败：输出差异过大")
                return False
                
        except Exception as e:
            print(f"验证过程出错: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        if self.torch_model is None:
            return {}
        
        info = {
            "input_shape": self.input_shape,
            "model_type": type(self.torch_model).__name__,
            "parameters": sum(p.numel() for p in self.torch_model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.torch_model.parameters() if p.requires_grad)
        }
        
        return info 