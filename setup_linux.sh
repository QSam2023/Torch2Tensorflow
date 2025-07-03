#!/bin/bash

# Linux环境 - PyTorch2TensorFlow 一键设置脚本
# 目标版本: Python 3.7 + TensorFlow 2.1.4 + PyTorch 1.13.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python 3.7
check_python() {
    print_info "检查Python 3.7..."
    
    if command -v python3.7 &> /dev/null; then
        PYTHON_CMD="python3.7"
        print_success "找到 python3.7"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
        if [ "$PYTHON_VERSION" = "3.7" ]; then
            PYTHON_CMD="python3"
            print_success "找到 python3 (版本 3.7)"
        else
            print_error "Python 3.7 未找到，当前版本: $PYTHON_VERSION"
            print_info "请先安装Python 3.7"
            exit 1
        fi
    else
        print_error "Python 3.7 未找到"
        print_info "请先安装Python 3.7"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    print_info "创建虚拟环境..."
    
    if [ -d "pytorch2tf_env" ]; then
        print_warning "虚拟环境已存在，是否删除重建？ (y/N)"
        read -r response
        if [[ $response =~ ^[Yy]$ ]]; then
            rm -rf pytorch2tf_env
            print_info "已删除旧的虚拟环境"
        else
            print_info "使用现有虚拟环境"
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv pytorch2tf_env
    print_success "虚拟环境创建成功"
}

# 激活虚拟环境
activate_venv() {
    print_info "激活虚拟环境..."
    source pytorch2tf_env/bin/activate
    
    # 验证Python版本
    CURRENT_PYTHON=$(python --version 2>&1)
    print_info "当前Python版本: $CURRENT_PYTHON"
}

# 升级pip
upgrade_pip() {
    print_info "升级pip..."
    python -m pip install --upgrade pip
    print_success "pip升级完成"
}

# 安装依赖
install_dependencies() {
    print_info "安装项目依赖..."
    
    if [ -f "requirements_linux.txt" ]; then
        print_info "使用 requirements_linux.txt 安装依赖..."
        
        # 分步安装以便于调试
        print_info "1. 安装numpy..."
        pip install "numpy>=1.19.5,<1.20.0"
        
        print_info "2. 安装PyTorch..."
        pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cpu
        
        print_info "3. 安装TensorFlow..."
        pip install tensorflow==2.1.3
        
        print_info "4. 安装ONNX相关..."
        pip install "onnx>=1.12.0,<1.15.0"
        pip install "onnxruntime>=1.12.0,<1.15.0"
        
        print_info "5. 安装转换工具..."
        pip install "tf2onnx>=1.9.0,<1.15.0"
        pip install "onnx-tf>=1.9.0,<1.11.0"
        
        print_info "6. 安装其他依赖..."
        pip install "protobuf>=3.20.0,<4.0.0"
        pip install "h5py>=2.10.0,<3.0.0"
        pip install "scipy>=1.5.0,<1.8.0"
        
    else
        print_warning "requirements_linux.txt 不存在，尝试使用 requirements.txt..."
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        else
            print_error "没有找到依赖文件"
            exit 1
        fi
    fi
    
    print_success "依赖安装完成"
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    if [ -f "test_installation_linux.py" ]; then
        python test_installation_linux.py
    else
        print_warning "Linux测试脚本不存在，使用通用测试脚本..."
        if [ -f "test_installation.py" ]; then
            python test_installation.py
        else
            print_warning "没有找到测试脚本，手动验证..."
            python -c "
import torch
import tensorflow as tf
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ TensorFlow: {tf.__version__}')
print('✓ 基本导入测试通过')
"
        fi
    fi
}

# 创建激活脚本
create_activation_script() {
    print_info "创建环境激活脚本..."
    
    cat > activate_linux.sh << 'EOF'
#!/bin/bash
# Linux环境激活脚本

# 进入项目目录
cd "$(dirname "$0")"

# 激活虚拟环境
if [ -d "pytorch2tf_env" ]; then
    source pytorch2tf_env/bin/activate
    echo "✅ PyTorch2TensorFlow Linux环境已激活"
    echo "Python版本: $(python --version)"
    echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
    echo "TensorFlow版本: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
    echo "当前目录: $(pwd)"
    echo ""
    echo "使用方法："
    echo "  python examples/convert_example.py  # 运行转换示例"
    echo "  python main.py --help              # 查看命令行选项"
    echo "  deactivate                         # 退出虚拟环境"
else
    echo "❌ 虚拟环境不存在，请先运行 setup_linux.sh"
    exit 1
fi
EOF
    
    chmod +x activate_linux.sh
    print_success "激活脚本创建完成: activate_linux.sh"
}

# 主函数
main() {
    echo "======================================================"
    echo "Linux环境 - PyTorch2TensorFlow 自动设置脚本"
    echo "目标版本: Python 3.7 + TensorFlow 2.1.4 + PyTorch 1.13.0"
    echo "======================================================"
    
    # 检查是否在项目根目录
    if [ ! -f "setup.py" ] || [ ! -d "converter" ]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    # 执行设置步骤
    check_python
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    verify_installation
    create_activation_script
    
    echo ""
    echo "======================================================"
    print_success "Linux环境设置完成！"
    echo ""
    echo "下一步操作："
    echo "1. 激活环境: source activate_linux.sh"
    echo "2. 运行测试: python test_installation_linux.py"
    echo "3. 运行示例: python examples/convert_example.py"
    echo "4. 使用工具: python main.py --help"
    echo ""
    echo "环境管理："
    echo "- 激活: source pytorch2tf_env/bin/activate"
    echo "- 退出: deactivate"
    echo "- 删除: rm -rf pytorch2tf_env"
    echo "======================================================"
}

# 允许以参数形式调用特定函数
if [ $# -eq 1 ]; then
    case $1 in
        "check")
            check_python
            ;;
        "create")
            create_venv
            ;;
        "install")
            activate_venv
            install_dependencies
            ;;
        "verify")
            activate_venv
            verify_installation
            ;;
        *)
            echo "用法: $0 [check|create|install|verify]"
            echo "或者直接运行: $0"
            ;;
    esac
else
    main
fi 