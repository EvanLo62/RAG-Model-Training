#!/bin/bash

# Download models from Google Drive

echo "Downloading Retriever & Reranker checkpoint..."

# Google Drive 檔案 ID
FILE_ID="1F21qZU50xsSty8ki4mci32IHUoq1oAv8"

# 使用 gdown 下載檔案
gdown --id ${FILE_ID} -O models.zip

# 檢查下載是否成功
if [ ! -f "models.zip" ]; then
    echo "Error: Failed to download models.zip"
    exit 1
fi

echo "Extracting models..."

# 解壓縮到當前目錄
unzip -q models.zip

# 檢查解壓縮是否成功
if [ ! -d "models" ]; then
    echo "Error: models directory not found after extraction"
    exit 1
fi

# 清理壓縮檔
rm models.zip

echo "Download and extraction completed successfully!"
echo "Files in models:"
ls -lh models/