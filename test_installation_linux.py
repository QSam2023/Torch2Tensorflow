#!/usr/bin/env python3
"""
Linux环境 - 项目安装和基本功能测试脚本 (TensorFlow 2.1.4)
"""

import sys
import os

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor != 7:
        print(f"⚠️  警告：建议使用Python 3.7，当前版本为 {version.major}.{version.minor}")
        return False
    else:
        print("✓ Python版本符合要求")
        return True

def test_imports():
    """测试模块导入"""
    print("\n测试模块导入...")
    import_results = {}
    
    # 测试PyTorch
    try:
        import torch
        torch_version = torch.__version__
        print(f"✓ PyTorch版本: {torch_version}")
        
        if torch_version.startswith('1.13'):
            import_results['torch'] = True
        else:
            print(f"⚠️  警告：PyTorch版本不匹配，期望1.13.x，实际{torch_version}")
            import_results['torch'] = False
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        import_results['torch'] = False
    
    # 测试TensorFlow
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"✓ TensorFlow版本: {tf_version}")
        
        if tf_version.startswith('2.1'):
            import_results['tensorflow'] = True
        else:
            print(f"⚠️  警告：TensorFlow版本不匹配，期望2.1.x，实际{tf_version}")
            import_results['tensorflow'] = False
    except ImportError as e:
        print(f"✗ TensorFlow导入失败: {e}")
        import_results['tensorflow'] = False
    
    # 测试ONNX
    try:
        import onnx
        print(f"✓ ONNX版本: {onnx.__version__}")
        import_results['onnx'] = True
    except ImportError as e:
        print(f"✗ ONNX导入失败: {e}")
        import_results['onnx'] = False
    
    # 测试ONNXRuntime
    try:
        import onnxruntime
        print(f"✓ ONNXRuntime版本: {onnxruntime.__version__}")
        import_results['onnxruntime'] = True
    except ImportError as e:
        print(f"✗ ONNXRuntime导入失败: {e}")
        import_results['onnxruntime'] = False
    
    # 测试onnx-tf
    try:
        import onnx_tf
        print(f"✓ onnx-tf版本: {onnx_tf.__version__}")
        import_results['onnx_tf'] = True
    except ImportError as e:
        print(f"✗ onnx-tf导入失败: {e}")
        import_results['onnx_tf'] = False
    
    # 测试自定义模块
    try:
        # 尝试使用TensorFlow 2.1.4兼容版本
        if os.path.exists('converter/torch_to_tf_v21.py'):
            sys.path.insert(0, '.')
            from converter.torch_to_tf_v21 import TorchToTensorFlowConverter
            from converter.model_utils import ModelUtils
            print("✓ 自定义转换器模块导入成功 (TensorFlow 2.1.4兼容版)")
            import_results['custom'] = True
        else:
            from converter import TorchToTensorFlowConverter, ModelUtils
            print("✓ 自定义转换器模块导入成功 (通用版)")
            import_results['custom'] = True
    except ImportError as e:
        print(f"✗ 自定义转换器模块导入失败: {e}")
        import_results['custom'] = False
    
    return all(import_results.values())

def test_tensorflow_compatibility():
    """测试TensorFlow 2.1.4特定功能"""
    print("\n测试TensorFlow 2.1.4兼容性...")
    try:
        import tensorflow as tf
        import numpy as np
        
        # 测试基本张量操作
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        print("✓ TensorFlow基本张量操作正常")
        
        # 测试SavedModel功能 (TensorFlow 2.1.4特有)
        @tf.function
        def simple_function(x):
            return tf.add(x, 1)
        
        # 创建测试输入
        test_input = tf.constant([1.0, 2.0, 3.0])
        result = simple_function(test_input)
        print("✓ TensorFlow函数装饰器正常")
        
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow兼容性测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    try:
        import torch
        import torch.nn as nn
        from converter.model_utils import ModelUtils
        
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

def test_conversion_pipeline():
    """测试转换流程"""
    print("\n测试转换流程...")
    try:
        import torch
        import tempfile
        import os
        
        # 选择合适的转换器
        if os.path.exists('converter/torch_to_tf_v21.py'):
            from converter.torch_to_tf_v21 import TorchToTensorFlowConverter
        else:
            from converter import TorchToTensorFlowConverter
        
        from converter.model_utils import ModelUtils
        
        # 创建简单的测试模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # 保存模型到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(model, tmp.name)
            model_path = tmp.name
        
        # 测试转换流程
        converter = TorchToTensorFlowConverter()
        converter.load_pytorch_model(model_path)
        converter.set_input_shape((1, 10))
        
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'test_model')
            success = converter.convert_via_onnx(output_path)
            
            if success:
                print("✓ 转换流程测试成功")
                result = True
            else:
                print("⚠️  转换流程完成，但可能存在问题")
                result = False
        
        # 清理临时文件
        os.unlink(model_path)
        
        return result
        
    except Exception as e:
        print(f"✗ 转换流程测试失败: {e}")
        return False

def print_environment_info():
    """打印环境信息"""
    print("\n=== 环境信息 ===")
    print(f"操作系统: {os.name}")
    print(f"Python路径: {sys.executable}")
    print(f"工作目录: {os.getcwd()}")
    
    # 检查虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ 运行在虚拟环境中")
    else:
        print("⚠️  未检测到虚拟环境")

def main():
    """主测试函数"""
    print("=" * 60)
    print("Linux环境 - Torch2Tensorflow 项目测试")
    print("目标版本: Python 3.7 + TensorFlow 2.1.4 + PyTorch 1.13.0")
    print("=" * 60)
    
    # 打印环境信息
    print_environment_info()
    
    # 运行所有测试
    tests = [
        ("Python版本检查", check_python_version),
        ("模块导入", test_imports),
        ("TensorFlow兼容性", test_tensorflow_compatibility),
        ("模型创建", test_model_creation),
        ("转换流程", test_conversion_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 测试通过")
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试出错: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！Linux环境配置成功。")
        print("\n下一步:")
        print("1. 运行示例: python examples/convert_example.py")
        print("2. 使用命令行: python main.py --help")
        print("3. 查看文档: README.md")
    else:
        print("❌ 部分测试失败，请检查安装。")
        print("\n建议:")
        print("1. 检查Python版本: python --version")
        print("2. 重新安装依赖: pip install -r requirements_linux.txt")
        print("3. 查看安装指南: INSTALL_LINUX.md")

if __name__ == "__main__":
    main() 