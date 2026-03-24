#!/bin/bash

# 设置变量
# 如果没有传入参数，默认文件名为 lab.cpp
SOURCE_FILE="${1:-lab.cpp}"
OUTPUT_DIR="build"
# 获取文件名（去掉后缀），用作可执行文件的名字
# 例如 lab.cpp -> lab
OUTPUT_NAME="${OUTPUT_DIR}/${SOURCE_FILE%.cpp}"

# 1. 运行 g++-15 进行编译
# -g 生成调试信息
# -O2 开启常用优化（适合算法竞赛）
# -std=c++20 使用 C++20 标准（可根据需要改为 c++17）
g++-15 -g -O2 -std=c++20 "$SOURCE_FILE" -o "$OUTPUT_NAME"

# 2. 检查编译是否成功
# $? 表示上一条命令的退出状态，0 代表成功
if [ $? -ne 0 ]; then
    echo "❌ 编译失败！请检查代码错误。"
    exit 1
fi

echo ""
echo "程序输出"
echo "----------------------------------------"

# 3. 运行生成的可执行文件
./"$OUTPUT_NAME"

echo ""
echo "----------------------------------------"
echo "🏁 程序运行结束。"
