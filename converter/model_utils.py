"""
模型工具类
"""

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any
import os


class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleMLPn(nn.Module):
    """简单的MLP模型"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLPn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ModelUtils:
    """模型相关工具函数"""
    
    @staticmethod
    def analyze_pytorch_model(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        分析PyTorch模型结构
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            
        Returns:
            Dict: 模型分析结果
        """
        model.eval()
        
        # 基本信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小（MB）
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = (param_size + buffer_size) / 1024 / 1024
        
        # 获取层信息
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                layers.append({
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                })
        
        # 测试前向传播
        try:
            dummy_input = torch.randn(input_shape)
            with torch.no_grad():
                output = model(dummy_input)
            output_shape = output.shape
            forward_pass = True
        except Exception as e:
            output_shape = None
            forward_pass = False
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "forward_pass_success": forward_pass,
            "layers": layers
        }
    
    @staticmethod
    def create_sample_models():
        """创建示例模型用于测试"""
        return {
            "SimpleCNN": SimpleCNN,
            "SimpleMLPn": SimpleMLPn
        }
    
    @staticmethod
    def save_pytorch_model(model: nn.Module, path: str, save_state_dict_only: bool = True):
        """
        保存PyTorch模型
        
        Args:
            model: PyTorch模型
            path: 保存路径
            save_state_dict_only: 是否只保存state_dict
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if save_state_dict_only:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)
        
        print(f"PyTorch模型已保存到: {path}")
    
    @staticmethod
    def validate_tensorflow_model(model_path: str, input_shape: Tuple[int, ...]) -> bool:
        """
        验证TensorFlow模型
        
        Args:
            model_path: TensorFlow模型路径
            input_shape: 输入形状
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 加载模型
            model = tf.saved_model.load(model_path)
            
            # 创建测试输入
            test_input = tf.random.normal(input_shape)
            
            # 进行推理
            output = model(test_input)
            
            print(f"TensorFlow模型验证成功")
            print(f"输入形状: {test_input.shape}")
            print(f"输出形状: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f"TensorFlow模型验证失败: {e}")
            return False
    
    @staticmethod
    def compare_model_outputs(pytorch_model: nn.Module, tf_model_path: str, 
                            input_shape: Tuple[int, ...], num_tests: int = 5) -> Dict[str, float]:
        """
        比较PyTorch和TensorFlow模型的输出
        
        Args:
            pytorch_model: PyTorch模型
            tf_model_path: TensorFlow模型路径
            input_shape: 输入形状
            num_tests: 测试次数
            
        Returns:
            Dict: 比较结果统计
        """
        try:
            # 加载TensorFlow模型
            tf_model = tf.saved_model.load(tf_model_path)
            pytorch_model.eval()
            
            differences = []
            
            for i in range(num_tests):
                # 生成随机输入
                test_input = np.random.randn(*input_shape).astype(np.float32)
                
                # PyTorch推理
                with torch.no_grad():
                    torch_input = torch.from_numpy(test_input)
                    torch_output = pytorch_model(torch_input).numpy()
                
                # TensorFlow推理
                tf_input = tf.constant(test_input)
                tf_output = tf_model(tf_input).numpy()
                
                # 计算差异
                diff = np.abs(torch_output - tf_output)
                differences.append({
                    "max_diff": np.max(diff),
                    "mean_diff": np.mean(diff),
                    "std_diff": np.std(diff)
                })
            
            # 统计结果
            max_diffs = [d["max_diff"] for d in differences]
            mean_diffs = [d["mean_diff"] for d in differences]
            
            result = {
                "num_tests": num_tests,
                "avg_max_diff": np.mean(max_diffs),
                "avg_mean_diff": np.mean(mean_diffs),
                "max_of_max_diffs": np.max(max_diffs),
                "std_of_max_diffs": np.std(max_diffs)
            }
            
            return result
            
        except Exception as e:
            print(f"模型输出比较失败: {e}")
            return {} 