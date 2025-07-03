# TensorFlow 版本更新总结：从 2.1.4 到 2.1.3

## 更新概述

本次更新将项目中所有的 TensorFlow 依赖版本从 2.1.4 调整为 2.1.3，并相应调整了所有相关依赖包的版本以确保兼容性。

## 主要更改

### 1. 依赖文件更新

#### requirements.txt
- `tensorflow==2.1.4` → `tensorflow==2.1.3`
- `numpy>=1.19.5` → `numpy>=1.19.2,<1.20.0`
- `onnx>=1.12.0` → `onnx>=1.4.0,<1.13.0`
- `onnxruntime>=1.12.0` → `onnxruntime>=1.0.0,<1.13.0`
- `tf2onnx>=1.9.0` → `tf2onnx>=1.5.0,<1.10.0`
- `protobuf>=3.20.0` → `protobuf>=3.8.0,<4.0.0`

#### requirements_linux.txt
- 与 requirements.txt 相同的版本更新
- 更新了 `onnx-tf>=1.9.0,<1.11.0` → `onnx-tf>=1.5.0,<1.10.0`
- 保持了 Linux 特定的依赖

#### requirements_mac_arm.txt
- 从 `tensorflow==2.13.0` 更新为 `tensorflow==2.1.3`
- 调整了所有相关依赖版本以匹配 TensorFlow 2.1.3

### 2. 代码文件更新

- **converter/__init__.py**: 更新版本检测逻辑注释
- **converter/torch_to_tf_v21.py**: 更新所有 TensorFlow 版本引用
- **test_installation_linux.py**: 更新测试脚本版本信息
- **setup_linux.sh**: 更新安装脚本版本

### 3. 文档文件更新

- **README_LINUX.md**: 更新所有版本信息
- **INSTALL_GUIDE.md**: 更新安装指南
- **QUICKSTART.md**: 更新快速开始版本要求
- **INSTALL_LINUX.md**: 更新 Linux 安装文档

## 版本兼容性调整

为确保与 TensorFlow 2.1.3 的完全兼容：

1. **NumPy**: `>=1.19.2,<1.20.0` (TensorFlow 2.1.3 不支持 NumPy 1.20+)
2. **Protobuf**: `>=3.8.0,<4.0.0` (需要较老的 protobuf 版本)
3. **ONNX**: `>=1.4.0,<1.13.0` (兼容版本范围)
4. **ONNXRuntime**: `>=1.0.0,<1.13.0` (与 ONNX 版本匹配)
5. **tf2onnx**: `>=1.5.0,<1.10.0` (支持 TensorFlow 2.1.3)
6. **onnx-tf**: `>=1.5.0,<1.10.0` (兼容 TensorFlow 2.1.3)

## 验证建议

1. 运行 `python test_installation.py` 验证安装
2. 运行 `python test_installation_linux.py` 验证 Linux 环境
3. 执行 `python examples/convert_example.py` 测试转换功能

## 注意事项

- Apple Silicon 用户：TensorFlow 2.1.3 对 Apple Silicon 支持有限
- 建议使用虚拟环境避免依赖冲突
- 确保 GPU 环境与 TensorFlow 2.1.3 兼容

---

**更新完成**：项目已成功从 TensorFlow 2.1.4 迁移至 2.1.3，所有相关文件已同步更新。
