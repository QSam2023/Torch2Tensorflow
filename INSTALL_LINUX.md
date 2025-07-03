# Linux环境安装指南

## 环境要求

- Ubuntu 18.04+ / CentOS 7+ / 其他Linux发行版
- Python 3.7 (精确版本要求)
- pip 19.0+
- Git

## 1. 安装Python 3.7

### Ubuntu/Debian系统

```bash
# 更新包列表
sudo apt update

# 安装Python 3.7和相关工具
sudo apt install python3.7 python3.7-dev python3.7-venv python3-pip

# 验证Python版本
python3.7 --version  # 应显示 Python 3.7.x
```

### CentOS/RHEL系统

```bash
# 安装EPEL仓库
sudo yum install epel-release

# 安装Python 3.7
sudo yum install python37 python37-devel python37-pip

# 创建符号链接
sudo ln -sf /usr/bin/python3.7 /usr/local/bin/python3.7
```

### 从源码编译（如果包管理器中没有Python 3.7）

```bash
# 安装编译依赖
sudo apt install build-essential libssl-dev libffi-dev python3-dev

# 下载并编译Python 3.7
wget https://www.python.org/ftp/python/3.7.16/Python-3.7.16.tgz
tar xzf Python-3.7.16.tgz
cd Python-3.7.16
./configure --enable-optimizations
make altinstall
sudo make altinstall
```

## 2. 创建虚拟环境

```bash
# 创建项目目录
mkdir -p ~/pytorch2tf
cd ~/pytorch2tf

# 克隆项目（如果还没有）
git clone <your-repo-url> .

# 创建虚拟环境
python3.7 -m venv pytorch2tf_env

# 激活虚拟环境
source pytorch2tf_env/bin/activate

# 验证Python版本
python --version  # 应显示 Python 3.7.x
```

## 3. 安装依赖

### 方法一：使用Linux专用requirements

```bash
# 激活虚拟环境
source pytorch2tf_env/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements_linux.txt
```

### 方法二：逐步安装（推荐，便于调试）

```bash
# 激活虚拟环境
source pytorch2tf_env/bin/activate

# 1. 安装基础数值计算库
pip install "numpy>=1.19.5,<1.20.0"

# 2. 安装PyTorch 1.13.0
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu

# 3. 安装TensorFlow 2.1.4
pip install tensorflow==2.1.4

# 4. 安装ONNX相关
pip install "onnx>=1.12.0,<1.15.0"
pip install "onnxruntime>=1.12.0,<1.15.0"

# 5. 安装转换工具
pip install "tf2onnx>=1.9.0,<1.15.0"
pip install "onnx-tf>=1.9.0,<1.11.0"

# 6. 安装其他依赖
pip install "protobuf>=3.20.0,<4.0.0"
pip install "h5py>=2.10.0,<3.0.0"
pip install "scipy>=1.5.0,<1.8.0"
```

## 4. 验证安装

```bash
# 激活环境
source pytorch2tf_env/bin/activate

# 运行测试脚本
python test_installation.py
```

预期输出：
```
✓ PyTorch版本: 1.13.0
✓ TensorFlow版本: 2.1.4
✓ 自定义转换器模块导入成功
🎉 所有测试通过！项目安装成功。
```

## 5. 运行示例

```bash
# 激活环境
source pytorch2tf_env/bin/activate

# 运行转换示例
python examples/convert_example.py
```

## 常见问题及解决方案

### 问题1：TensorFlow 2.1.4安装失败

```bash
# 尝试指定特定的源
pip install tensorflow==2.1.4 -i https://pypi.douban.com/simple/

# 或者安装CPU版本
pip install tensorflow-cpu==2.1.4
```

### 问题2：ONNX版本冲突

```bash
# 卸载现有版本
pip uninstall onnx onnxruntime tf2onnx onnx-tf

# 重新安装兼容版本
pip install onnx==1.12.0 onnxruntime==1.12.0 tf2onnx==1.9.0 onnx-tf==1.9.0
```

### 问题3：protobuf版本冲突

```bash
# 安装兼容的protobuf版本
pip install protobuf==3.20.3
```

### 问题4：NumPy版本不兼容

```bash
# TensorFlow 2.1.4需要特定的NumPy版本
pip install "numpy>=1.19.5,<1.20.0"
```

## 6. 性能优化（可选）

### 安装GPU版本PyTorch（如果有CUDA）

```bash
# 检查CUDA版本
nvidia-smi

# 安装对应的PyTorch GPU版本
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 安装GPU版本TensorFlow（如果有CUDA）

```bash
# TensorFlow 2.1.4的GPU版本
pip install tensorflow-gpu==2.1.4
```

## 7. 环境管理脚本

创建便捷的环境管理脚本：

```bash
# 创建激活脚本
cat > activate_env.sh << 'EOF'
#!/bin/bash
cd ~/pytorch2tf
source pytorch2tf_env/bin/activate
echo "✅ PyTorch2TensorFlow环境已激活"
echo "Python版本: $(python --version)"
echo "当前目录: $(pwd)"
EOF

chmod +x activate_env.sh
```

使用方法：
```bash
# 激活环境
source activate_env.sh

# 或者直接执行
./activate_env.sh && bash
```

## 版本对比

| 组件 | Mac版本 | Linux版本 | 说明 |
|------|---------|-----------|------|
| Python | 3.8 | **3.7** | 恢复原始要求 |
| TensorFlow | 2.13.0 | **2.1.4** | 恢复原始要求 |
| PyTorch | 1.13.0 | **1.13.0** | 保持一致 |
| 环境管理 | conda | **venv** | 使用Python标准库 |

## 测试清单

- [ ] Python 3.7安装成功
- [ ] 虚拟环境创建成功  
- [ ] 所有依赖安装无冲突
- [ ] `test_installation.py` 测试通过
- [ ] 示例转换运行成功
- [ ] 模型文件正确生成 