#!/bin/bash
command -v rclone >/dev/null 2>&1 || {
    echo "❌ エラー: rclone がインストールされていません。"
    exit 1
}

echo "🚀 イヴの観測データをGoogle Driveへ転送開始..."

# rclone copy を使用
rclone copy ~/project_eve gdrive:ProjectEve_Sync \
  --progress \
  --exclude "eve_env/**" \
  --exclude "__pycache__/**" \
  --exclude ".git/**" \
  --exclude "*.pyc" \
  --exclude ".jax_cache/**" \
  --exclude "*.png" \
  --exclude "*.mp4"

echo "✅ 転送完了。AIとのコンテキスト同期が可能です。"
