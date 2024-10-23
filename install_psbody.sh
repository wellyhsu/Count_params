#!/bin/bash
## install psbody ##

# 在任何命令失敗時停止執行
set -e

# 進入mesh folder
cd mesh || { echo "Failed to enter mesh directory."; exit 1; }

# install psbody-mesh
pip install --no-deps --verbose --no-cache-dir .

# 確認操作已完成
echo "Operation completed."

cd ../HandMesh