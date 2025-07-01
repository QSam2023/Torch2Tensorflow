#!/usr/bin/env python3
"""
项目安装和基本功能测试脚本
"""

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        import tensorflow as tf
        print(f"✓ TensorFlow版本: {tf.__version__}")
        
        import onnx
        print(f"✓ ONNX版本: {onnx.__version__}")
        
        # 测试我们的模块
        from converter import TorchToTensorFlowConverter, ModelUtils
        print("✓ 自定义转换器模块导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    try:
        from converter import TorchToTensorFlowConverter, ModelUtils
        
        # 创建转换器
        converter = TorchToTensorFlowConverter()
        print("✓ 转换器创建成功")
        
        # 测试工具类
        sample_models = ModelUtils.create_sample_models()
        print("✓ 示例模型类创建成功")
        print(f"  可用模型: {list(sample_models.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    try:
        import torch
        import torch.nn as nn
        from converter import ModelUtils
        
        # 创建示例模型
        sample_models = ModelUtils.create_sample_models()
        
        # 测试CNN模型
        cnn_model = sample_models["SimpleCNN"](num_classes=10)
        print("✓ CNN模型创建成功")
        
        # 测试模型分析
        input_shape = (1, 3, 32, 32)
        model_info = ModelUtils.analyze_pytorch_model(cnn_model, input_shape)
        print(f"✓ 模型分析成功")
        print(f"  参数数量: {model_info['total_parameters']}")
        print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Torch2Tensorflow 项目测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("基本功能", test_basic_functionality),
        ("模型创建", test_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
            print(f"✓ {test_name} 测试通过")
        else:
            print(f"✗ {test_name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目安装成功。")
        print("\n下一步:")
        print("1. 运行示例: python examples/convert_example.py")
        print("2. 使用命令行: python main.py --help")
        print("3. 查看文档: README.md")
    else:
        print("❌ 部分测试失败，请检查依赖安装。")
        print("建议运行: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 