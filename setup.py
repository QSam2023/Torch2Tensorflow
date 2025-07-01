#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="torch2tensorflow",
    version="1.0.0",
    author="Torch2Tensorflow Team",
    author_email="your-email@example.com",
    description="A tool for converting PyTorch models to TensorFlow models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Torch2Tensorflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "torch2tf=main:main",
        ],
    },
    keywords="pytorch tensorflow model conversion onnx deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Torch2Tensorflow/issues",
        "Source": "https://github.com/yourusername/Torch2Tensorflow",
    },
) 