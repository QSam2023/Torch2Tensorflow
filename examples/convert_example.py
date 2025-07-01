#!/usr/bin/env python3
"""
转换示例脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from converter import TorchToTensorFlowConverter, ModelUtils


def create_and_convert_sample_model():
    """创建并转换示例模型"""
    
    # 获取示例模型
    sample_models = ModelUtils.create_sample_models()
    
    # 创建简单CNN模型
    print("=== 创建示例CNN模型 ===")
    model = sample_models["SimpleCNN"](num_classes=10)
    
    # 随机初始化权重
    model.apply(lambda m: nn.init.xavier_uniform_(m.weight) 
                if isinstance(m, (nn.Linear, nn.Conv2d)) else None)
    
    # 分析模型
    input_shape = (1, 3, 32, 32)
    model_info = ModelUtils.analyze_pytorch_model(model, input_shape)
    
    print("模型信息:")
    for key, value in model_info.items():
        if key != 'layers':  # 跳过详细层信息
            print(f"  {key}: {value}")
    
    # 保存PyTorch模型
    pytorch_model_path = "models/sample_cnn.pth"
    os.makedirs("models", exist_ok=True)
    ModelUtils.save_pytorch_model(model, pytorch_model_path)
    
    # 创建转换器
    print("\n=== 开始转换 ===")
    converter = TorchToTensorFlowConverter()
    
    # 加载模型
    converter.load_pytorch_model(pytorch_model_path, model_class=None)
    converter.set_input_shape(input_shape)
    
    # 转换模型
    tf_model_path = "models/sample_cnn_tf"
    success = converter.convert_via_onnx(tf_model_path)
    
    if success:
        print("✓ 模型转换成功")
        
        # 验证转换
        print("\n=== 验证转换结果 ===")
        if converter.validate_conversion(tf_model_path):
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")
            
        # 详细比较
        print("\n=== 详细输出比较 ===")
        comparison = ModelUtils.compare_model_outputs(
            model, tf_model_path, input_shape, num_tests=5
        )
        
        if comparison:
            print("比较结果:")
            for key, value in comparison.items():
                print(f"  {key}: {value:.8f}")
    else:
        print("✗ 模型转换失败")


def create_and_convert_mlp_model():
    """创建并转换MLP模型示例"""
    
    print("\n" + "="*50)
    print("=== 创建示例MLP模型 ===")
    
    # 获取示例模型
    sample_models = ModelUtils.create_sample_models()
    
    # 创建MLP模型
    model = sample_models["SimpleMLPn"](input_size=784, hidden_size=256, num_classes=10)
    
    # 分析模型
    input_shape = (1, 784)
    model_info = ModelUtils.analyze_pytorch_model(model, input_shape)
    
    print("MLP模型信息:")
    for key, value in model_info.items():
        if key != 'layers':
            print(f"  {key}: {value}")
    
    # 保存模型
    pytorch_model_path = "models/sample_mlp.pth"
    ModelUtils.save_pytorch_model(model, pytorch_model_path)
    
    # 转换模型
    print("\n=== 转换MLP模型 ===")
    converter = TorchToTensorFlowConverter()
    converter.load_pytorch_model(pytorch_model_path, model_class=None)
    converter.set_input_shape(input_shape)
    
    tf_model_path = "models/sample_mlp_tf"
    success = converter.convert_via_onnx(tf_model_path)
    
    if success:
        print("✓ MLP模型转换成功")
        
        # 验证
        if converter.validate_conversion(tf_model_path):
            print("✓ MLP模型验证通过")
    else:
        print("✗ MLP模型转换失败")


if __name__ == "__main__":
    print("Torch2Tensorflow 转换示例")
    print("="*50)
    
    try:
        # 转换CNN模型
        create_and_convert_sample_model()
        
        # 转换MLP模型
        create_and_convert_mlp_model()
        
        print("\n" + "="*50)
        print("所有示例转换完成！")
        print("生成的文件:")
        print("  - models/sample_cnn.pth (PyTorch CNN)")
        print("  - models/sample_cnn_tf/ (TensorFlow CNN)")
        print("  - models/sample_mlp.pth (PyTorch MLP)")
        print("  - models/sample_mlp_tf/ (TensorFlow MLP)")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc() 