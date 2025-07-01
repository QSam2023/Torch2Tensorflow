# Torch2Tensorflow

一个用于将PyTorch模型转换为TensorFlow模型的Python工具库。

## 功能特点

- 支持PyTorch 1.13.0 到TensorFlow 2.1.x的模型转换
- 基于ONNX中间格式进行转换，确保兼容性
- 提供模型验证功能，确保转换准确性
- 支持多种模型架构（CNN、MLP等）
- 命令行工具和Python API两种使用方式
- 详细的转换过程日志和错误提示

## 安装要求

### 系统要求
- Python 3.7+
- PyTorch 1.13.0
- TensorFlow 2.1.x

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 命令行使用

```bash
# 基本转换
python main.py --input model.pth --output tf_model/ --input-shape 1 3 224 224

# 带验证的转换
python main.py --input model.pth --output tf_model/ --input-shape 1 3 224 224 --validate

# 分析模型并转换
python main.py --input model.pth --output tf_model/ --input-shape 1 3 224 224 --analyze --validate
```

### 2. Python API使用

```python
from converter import TorchToTensorFlowConverter

# 创建转换器
converter = TorchToTensorFlowConverter()

# 加载PyTorch模型
converter.load_pytorch_model("model.pth")

# 设置输入形状
converter.set_input_shape((1, 3, 224, 224))

# 执行转换
success = converter.convert_via_onnx("tf_model/")

# 验证转换结果
if success:
    converter.validate_conversion("tf_model/")
```

### 3. 运行示例

```bash
python examples/convert_example.py
```

这将创建示例模型并进行转换演示。

## 转换流程

1. **加载PyTorch模型**: 支持`.pth`文件和模型类构建
2. **PyTorch → ONNX**: 使用torch.onnx.export转换为ONNX格式
3. **ONNX → TensorFlow**: 使用tf2onnx转换为TensorFlow SavedModel格式
4. **验证转换**: 比较PyTorch和TensorFlow模型的输出一致性

## 支持的模型类型

- 卷积神经网络(CNN)
- 多层感知机(MLP)
- 残差网络(ResNet)
- 自定义模型架构

## API 参考

### TorchToTensorFlowConverter

主要的转换器类。

#### 方法

- `load_pytorch_model(model_path, model_class=None, **kwargs)`: 加载PyTorch模型
- `set_input_shape(input_shape)`: 设置输入张量形状
- `convert_via_onnx(output_path, input_names=None, output_names=None, opset_version=11)`: 通过ONNX转换
- `validate_conversion(tf_model_path, test_inputs=None)`: 验证转换结果
- `get_model_info()`: 获取模型信息

### ModelUtils

模型工具类。

#### 静态方法

- `analyze_pytorch_model(model, input_shape)`: 分析PyTorch模型
- `create_sample_models()`: 创建示例模型
- `save_pytorch_model(model, path, save_state_dict_only=True)`: 保存PyTorch模型
- `validate_tensorflow_model(model_path, input_shape)`: 验证TensorFlow模型
- `compare_model_outputs(pytorch_model, tf_model_path, input_shape, num_tests=5)`: 比较模型输出

## 命令行参数

- `--input, -i`: 输入的PyTorch模型文件路径
- `--output, -o`: 输出的TensorFlow模型目录路径
- `--input-shape`: 输入张量形状
- `--input-names`: 输入节点名称
- `--output-names`: 输出节点名称
- `--opset-version`: ONNX opset版本
- `--validate`: 验证转换结果
- `--analyze`: 分析模型结构

## 常见问题

### Q: 转换失败怎么办？

A: 检查以下项目：
1. PyTorch模型是否能正常加载
2. 输入形状是否正确
3. 模型是否包含不支持的操作
4. ONNX opset版本是否兼容

### Q: 转换后精度损失怎么办？

A: 尝试：
1. 调整ONNX opset版本
2. 检查模型中的数值稳定性
3. 使用`--validate`参数检查差异

### Q: 支持哪些PyTorch操作？

A: 支持大部分常见操作，包括：
- 卷积、池化、全连接层
- 激活函数（ReLU、Sigmoid、Tanh等）
- 批归一化、Dropout
- 基本数学运算

## 项目结构

```
Torch2Tensorflow/
├── converter/              # 转换器模块
│   ├── __init__.py
│   ├── torch_to_tf.py      # 主转换器
│   └── model_utils.py      # 工具函数
├── examples/               # 示例代码
│   └── convert_example.py
├── main.py                 # 命令行入口
├── requirements.txt        # 依赖要求
├── README.md              # 项目文档
└── LICENSE                # 许可证
```

## 版本支持

- PyTorch: 1.13.0
- TensorFlow: 2.1.x
- ONNX: 1.12.0+
- Python: 3.7+

## 许可证

本项目基于MIT许可证开源。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基本的PyTorch到TensorFlow转换
- 提供命令行工具和Python API
- 包含模型验证功能 