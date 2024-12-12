#!/bin/bash

# 脚本名称: make_executable.sh
# 功能: 为当前目录下的所有 .sh 文件赋予执行权限

# 获取当前目录下的所有 .sh 文件
sh_files=(*.sh)

# 检查是否存在 .sh 文件
if [ "${sh_files[0]}" == "*.sh" ]; then
    echo "当前目录下没有找到 .sh 文件。"
    exit 0
fi

# 遍历每个 .sh 文件并赋予执行权限
for file in "${sh_files[@]}"; do
    if [ -f "$file" ]; then
        chmod +x "$file"
        echo "已赋予执行权限: $file"
    else
        echo "跳过非文件: $file"
    fi
done

echo "所有 .sh 文件的执行权限已更新。"
