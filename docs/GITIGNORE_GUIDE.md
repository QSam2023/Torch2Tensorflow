# Git忽略文件配置说明

## 配置概览

本项目的`.gitignore`文件包含了以下类别的忽略配置：

### 🐍 Python相关
- `__pycache__/` - Python字节码缓存
- `*.pyc`, `*.pyo` - 编译的Python文件
- `*.egg-info/` - Python包信息
- `venv/`, `env/` - 虚拟环境目录

### 🧠 深度学习模型文件
- `*.pth`, `*.pt` - PyTorch模型文件
- `*.pb`, `*.ckpt*` - TensorFlow模型文件
- `*.onnx` - ONNX模型文件
- `models/*/` - 模型目录（保留.gitkeep）

### 📊 数据文件
- `*.csv`, `*.json` - 数据集文件
- `*.pkl`, `*.pickle` - Python序列化文件
- `*.npy`, `*.npz` - NumPy数组文件
- `data/`, `datasets/` - 数据目录

### 💻 IDE和编辑器
- `.vscode/` - VS Code配置
- `.idea/` - PyCharm配置
- `*.sublime-*` - Sublime Text配置

### 🖥️ 操作系统
- `.DS_Store` - macOS文件系统
- `Thumbs.db` - Windows缩略图
- `*~` - Linux临时文件

### 📝 日志和缓存
- `*.log` - 日志文件
- `.cache/` - 缓存目录
- `logs/`, `runs/` - 训练日志

## 当前忽略的文件

运行 `git status --ignored` 可以查看当前被忽略的文件：

```bash
$ git status --ignored
Ignored files:
        converter/__pycache__/     # Python缓存
        models/sample_cnn.pth      # PyTorch模型
        models/sample_cnn_tf/      # TensorFlow模型目录
        models/sample_mlp.pth      # PyTorch模型
        models/sample_mlp_tf/      # TensorFlow模型目录
```

## 特殊配置

### models目录
- `models/*/` - 忽略所有子目录
- `!models/.gitkeep` - 保留.gitkeep文件以维持目录结构
- 可以通过取消注释 `!models/sample_*` 来跟踪示例模型

### 配置文件
- 忽略可能包含敏感信息的配置文件
- `.env*`, `config.*`, `secrets.*`

## 添加强制跟踪

如果需要强制添加被忽略的文件：

```bash
git add -f models/specific_model.pth
```

## 检查忽略状态

```bash
# 查看所有文件状态（包括被忽略的）
git status --ignored

# 检查特定文件是否被忽略
git check-ignore models/sample_cnn.pth
``` 