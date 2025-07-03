# 安装指南 - ARM Mac (Apple Silicon)

## 环境要求

- macOS 11.0+ (Apple Silicon)
- Python 3.8+ (Python 3.7 在ARM Mac上不可用)
- conda 或 miniconda

## 1. 创建Conda环境

```bash
# 创建环境（使用Python 3.8，因为3.7不可用）
conda create -n pytorch2tf python=3.8 -y

# 激活环境
conda activate pytorch2tf
```

## 2. 安装依赖

### 方法一：使用ARM Mac专用requirements文件

```bash
pip install -r requirements_mac_arm.txt
```

### 方法二：逐步安装

```bash
# 核心深度学习框架
pip install torch==1.13.0
pip install tensorflow==2.13.0

# ONNX相关
pip install "onnx>=1.12.0"
pip install "onnxruntime>=1.12.0"

# 转换工具
pip install "tf2onnx>=1.9.0"
pip install onnx-tf
pip install tensorflow-probability

# 其他依赖
pip install "numpy>=1.19.5"
pip install "protobuf>=3.20.0"
```

## 3. 验证安装

```bash
python test_installation.py
```

预期输出：
```
🎉 所有测试通过！项目安装成功。
```

## 4. 运行示例

```bash
python examples/convert_example.py
```

## 版本说明

### 与原始要求的差异

| 组件 | 原始要求 | ARM Mac实际版本 | 原因 |
|------|----------|----------------|------|
| Python | 3.7 | 3.8 | Python 3.7在ARM Mac conda中不可用 |
| TensorFlow | 2.1.x | 2.13.0 | TensorFlow 2.1.x不支持Apple Silicon |

### 核心功能兼容性

✅ PyTorch模型加载和导出
✅ ONNX中间转换
✅ TensorFlow模型生成
✅ 模型验证和比较
✅ 命令行工具
✅ Python API

## 故障排除

### 问题1：tensorflow_probability缺失
```bash
pip install tensorflow-probability
```

### 问题2：onnx-tf缺失
```bash
pip install onnx-tf
```

### 问题3：版本冲突
删除环境重新创建：
```bash
conda remove -n pytorch2tf --all
```
然后重新执行安装步骤。

## 成功验证

安装成功后，您应该能看到：

1. **测试通过**：`python test_installation.py` 显示所有测试通过
2. **示例运行**：`python examples/convert_example.py` 成功转换模型
3. **模型文件生成**：在 `models/` 目录下生成PyTorch和TensorFlow模型文件

生成的文件结构：
```
models/
├── sample_cnn.pth          # PyTorch CNN模型
├── sample_cnn_tf/          # TensorFlow CNN模型
│   ├── saved_model.pb
│   ├── variables/
│   └── ...
├── sample_mlp.pth          # PyTorch MLP模型
└── sample_mlp_tf/          # TensorFlow MLP模型
    ├── saved_model.pb
    ├── variables/
    └── ...
``` 