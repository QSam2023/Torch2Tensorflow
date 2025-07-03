# Linux环境部署指南

## 概述

本项目现已完全支持Linux环境，使用原始要求的版本配置：
- **Python 3.7** (精确版本要求)
- **TensorFlow 2.1.3** (与已有模型兼容)
- **PyTorch 1.13.0** (保持一致)
- **虚拟环境** (venv，不使用conda)

## 🚀 快速开始

### 一键安装（推荐）

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd Torch2Tensorflow

# 2. 运行自动安装脚本
./setup_linux.sh

# 3. 激活环境
source activate_linux.sh

# 4. 运行测试
python test_installation_linux.py
```

### 手动安装

如果自动脚本遇到问题，可以手动执行：

```bash
# 1. 安装Python 3.7
sudo apt install python3.7 python3.7-dev python3.7-venv

# 2. 创建虚拟环境
python3.7 -m venv pytorch2tf_env
source pytorch2tf_env/bin/activate

# 3. 安装依赖
pip install -r requirements_linux.txt

# 4. 验证安装
python test_installation_linux.py
```

## 📁 文件说明

### Linux专用文件

| 文件 | 说明 |
|------|------|
| `requirements_linux.txt` | Linux环境专用依赖列表 |
| `INSTALL_LINUX.md` | 详细安装指南 |
| `test_installation_linux.py` | Linux环境测试脚本 |
| `setup_linux.sh` | 一键安装脚本 |
| `activate_linux.sh` | 环境激活脚本（自动生成） |
| `converter/torch_to_tf_v21.py` | TensorFlow 2.1.3兼容转换器 |

### 智能版本选择

项目会自动检测TensorFlow版本并选择合适的转换器：

```python
from converter import TorchToTensorFlowConverter  # 自动选择版本
```

- TensorFlow 2.1.x → 使用 `torch_to_tf_v21.py`
- TensorFlow 2.13.x → 使用 `torch_to_tf.py`

## 🔧 环境管理

### 常用命令

```bash
# 激活环境
source pytorch2tf_env/bin/activate
# 或者
source activate_linux.sh

# 查看环境信息
python -c "from converter import get_environment_info; print(get_environment_info())"

# 退出环境
deactivate

# 删除环境（重新安装时）
rm -rf pytorch2tf_env
```

### 部分安装（调试用）

```bash
# 只检查Python版本
./setup_linux.sh check

# 只创建虚拟环境
./setup_linux.sh create

# 只安装依赖
./setup_linux.sh install

# 只验证安装
./setup_linux.sh verify
```

## 🧪 测试和验证

### 运行测试

```bash
# 完整测试
python test_installation_linux.py

# 快速验证
python -c "
import torch, tensorflow as tf
from converter import TorchToTensorFlowConverter, ModelUtils
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ TensorFlow: {tf.__version__}')
print('✓ 所有模块导入成功')
"
```

### 转换示例

```bash
# 运行转换示例
python examples/convert_example.py

# 使用命令行工具
python main.py convert --input models/sample_cnn.pth --output models/sample_cnn_tf_linux
```

## 🐛 故障排除

### 常见问题

#### 1. Python 3.7 不可用

```bash
# Ubuntu 18.04+
sudo apt update
sudo apt install python3.7 python3.7-dev python3.7-venv

# CentOS 7+
sudo yum install epel-release
sudo yum install python37 python37-devel

# 从源码编译（最后手段）
wget https://www.python.org/ftp/python/3.7.16/Python-3.7.16.tgz
tar xzf Python-3.7.16.tgz
cd Python-3.7.16
./configure --enable-optimizations
make altinstall
```

#### 2. TensorFlow 2.1.4 安装失败

```bash
# 使用国内镜像源
pip install tensorflow==2.1.4 -i https://pypi.douban.com/simple/

# 或者使用CPU版本
pip install tensorflow-cpu==2.1.4
```

#### 3. 版本冲突

```bash
# 完全重新安装
rm -rf pytorch2tf_env
./setup_linux.sh
```

#### 4. 权限问题

```bash
# 给脚本添加执行权限
chmod +x setup_linux.sh activate_linux.sh

# 如果需要sudo权限安装Python
sudo ./setup_linux.sh  # 不推荐，建议使用用户权限
```

## 📊 版本对比

| 环境 | Python | TensorFlow | PyTorch | 环境管理 | 状态 |
|------|--------|------------|---------|----------|------|
| **Mac (ARM)** | 3.8 | 2.13.0 | 1.13.0 | conda | ✅ 已验证 |
| **Linux** | **3.7** | **2.1.3** | 1.13.0 | venv | ✅ 新增 |

## 🎯 使用建议

### 开发环境

- **Mac用户**：使用conda环境，版本为Python 3.8 + TensorFlow 2.13.0
- **Linux用户**：使用venv环境，版本为Python 3.7 + TensorFlow 2.1.3
- **生产环境**：建议使用Linux环境以确保与已有模型的最佳兼容性

### 模型兼容性

- **新模型开发**：可以使用Mac环境（TensorFlow 2.13.0）
- **已有模型转换**：使用Linux环境（TensorFlow 2.1.3）确保兼容性
- **跨平台部署**：项目自动检测版本，无需修改代码

## 📞 支持

如果遇到问题：

1. 查看详细安装指南：`INSTALL_LINUX.md`
2. 运行诊断脚本：`python test_installation_linux.py`
3. 查看环境信息：`from converter import get_environment_info; print(get_environment_info())`
4. 检查git状态确认所有Linux文件都已正确添加

---

**项目现已完全支持Linux环境！可以在Linux系统中使用原始要求的版本进行模型转换。** 