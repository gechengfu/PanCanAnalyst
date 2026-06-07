#!/bin/bash
# PanCanAnalyst 启动脚本
# 用法: bash start.sh

cd "$(dirname "$0")"

echo "=== PanCanAnalyst 胰腺癌多组学分析平台 ==="
echo ""

# 检查 Python 依赖
echo "[1/2] 检查依赖..."
pip install -q flask flask-cors requests pandas numpy scipy 2>/dev/null

# 启动 Flask 服务
echo "[2/2] 启动服务器..."
echo ""
echo "  浏览器访问: http://127.0.0.1:5001"
echo "  按 Ctrl+C 停止服务"
echo ""

python3 app.py
