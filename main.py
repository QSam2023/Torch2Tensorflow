#!/usr/bin/env python3
"""
Torch2Tensorflow 主程序
"""

import argparse
import os
import sys
from converter import TorchToTensorFlowConverter, ModelUtils


def main():
    parser = argparse.ArgumentParser(description="PyTorch模型转TensorFlow模型工具")
    
    parser.add_argument("--input", "-i", required=True, 
                       help="输入的PyTorch模型文件路径")
    parser.add_argument("--output", "-o", required=True,
                       help="输出的TensorFlow模型目录路径")
    parser.add_argument("--input-shape", required=True, nargs="+", type=int,
                       help="输入张量形状，例如: 1 3 224 224")
    parser.add_argument("--input-names", nargs="+", default=["input"],
                       help="输入节点名称")
    parser.add_argument("--output-names", nargs="+", default=["output"],
                       help="输出节点名称")
    parser.add_argument("--opset-version", type=int, default=11,
                       help="ONNX opset版本")
    parser.add_argument("--validate", action="store_true",
                       help="验证转换结果")
    parser.add_argument("--analyze", action="store_true",
                       help="分析模型结构")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建转换器
    converter = TorchToTensorFlowConverter()
    
    # 加载PyTorch模型
    print("加载PyTorch模型...")
    if not converter.load_pytorch_model(args.input):
        print("模型加载失败")
        sys.exit(1)
    
    # 设置输入形状
    input_shape = tuple(args.input_shape)
    converter.set_input_shape(input_shape)
    
    # 分析模型（如果需要）
    if args.analyze:
        print("\n=== 模型分析 ===")
        info = converter.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    # 进行转换
    print("\n开始转换模型...")
    success = converter.convert_via_onnx(
        args.output,
        input_names=args.input_names,
        output_names=args.output_names,
        opset_version=args.opset_version
    )
    
    if not success:
        print("模型转换失败")
        sys.exit(1)
    
    # 验证转换结果（如果需要）
    if args.validate:
        print("\n验证转换结果...")
        if converter.validate_conversion(args.output):
            print("转换验证成功✓")
        else:
            print("转换验证失败✗")
            sys.exit(1)
    
    print(f"\n转换完成！TensorFlow模型已保存到: {args.output}")


if __name__ == "__main__":
    main() 