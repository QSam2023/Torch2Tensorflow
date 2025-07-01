# 快速开始指南

## 1. 安装依赖

```bash
# 安装所需依赖
pip install -r requirements.txt
```

## 2. 验证安装

```bash
# 运行测试脚本
python test_installation.py
```

## 3. 运行示例

```bash
# 运行完整示例转换
python examples/convert_example.py
```

这将会：
- 创建示例CNN和MLP模型
- 将PyTorch模型转换为TensorFlow模型
- 验证转换结果的准确性
- 保存模型到 `models/` 目录

## 4. 命令行使用

```bash
# 查看帮助
python main.py --help

# 基本转换（需要先有一个PyTorch模型文件）
python main.py --input your_model.pth --output tf_model/ --input-shape 1 3 224 224

# 带验证的转换
python main.py --input your_model.pth --output tf_model/ --input-shape 1 3 224 224 --validate

# 分析模型并转换
python main.py --input your_model.pth --output tf_model/ --input-shape 1 3 224 224 --analyze --validate
```

## 5. Python API使用

```python
from converter import TorchToTensorFlowConverter

# 创建转换器
converter = TorchToTensorFlowConverter()

# 加载PyTorch模型
converter.load_pytorch_model("your_model.pth")

# 设置输入形状
converter.set_input_shape((1, 3, 224, 224))

# 执行转换
success = converter.convert_via_onnx("tf_model/")

# 验证转换结果
if success:
    converter.validate_conversion("tf_model/")
```

## 支持的版本

- PyTorch: 1.13.0
- TensorFlow: 2.1.4
- Python: 3.7+

## 常见问题

### Q: 如何准备PyTorch模型？

A: 确保您的模型：
1. 可以成功加载
2. 处于eval模式
3. 支持您指定的输入形状

### Q: 转换失败怎么办？

A: 检查：
1. 依赖是否正确安装
2. 输入形状是否正确
3. 模型是否包含不支持的操作

### Q: 如何验证转换质量？

A: 使用 `--validate` 参数，工具会自动比较PyTorch和TensorFlow模型的输出差异。 