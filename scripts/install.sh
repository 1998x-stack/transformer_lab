#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt
# 可选：加速 HuggingFace 下载（国内可配镜像/代理）
