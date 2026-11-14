# Makefile — 构建 **无 Python 依赖** 的 C ABI 动态库与 CLI
#
# 目的（对应 docs/PROJECT.md 的"通用性"目标）：
#   让本项目能在**没有 Python 的机器**上构建与运行，从而作为基础设施底层
#   模块被任何语言调用（Rust/Go/Java/Julia/C#/C）。判据是"能在这种机器上
#   跑下来"——因此本文件**不调用 python、不引用 pybind11**。
#
# 与另一条构建路径的关系：
#   bash cpp/build_nano.sh   ← Python 扩展（pybind11），供 kakeya_ai 包使用
#   make -f cpp/Makefile     ← 本文件，C ABI，供任意语言使用
#   两者共用同一份几何算法语义，产物不同，互不干扰。
#
# 目录结构（2026-11-06 按构建路径整理）：
#   cpp/kernel/  几何内核本体（geo_core.hpp / parallel.hpp / 两个 kernel.cpp）
#                 ——pybind11 扩展用，编译期依赖 Python 头文件
#   cpp/abi/     C ABI 稳定接口（ka_abi.h / ka_abi.cpp / ka_cli.cpp / test_abi.c）
#                 ——**无 Python 依赖**，目标是能在没装 Python 的机器上编译
#   两条路径共用 cpp/kernel/geo_core.hpp 的几何实现（唯一真源）。
#
# 常用：
#   make -f cpp/Makefile                  # 构建库 + CLI（默认 -O3）
#   make -f cpp/Makefile check            # 用 C 版测试程序验证语义
#   make -f cpp/Makefile clean
#
# 交叉编译示例（在 x86_64 上出 aarch64 库）：
#   make -f cpp/Makefile CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++
#
# 指令集可移植性（见 docs/PROJECT.md 兼容性一节）：
#   默认**不加** -march=native —— 那会让产物只能在构建机上跑（换台机器
#   报 Illegal instruction）。要极致性能可在本机构建时显式覆盖：
#     make -f cpp/Makefile MARCH="-march=native"

CXX      ?= c++
CC       ?= cc
# 基线指令集：x86-64-v2 是 2009 年后 CPU 的通用下限（SSE4.2/POPCNT）。
# 留成变量是为了让调用方能按目标平台覆盖。
MARCH    ?=
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC -fvisibility=hidden $(MARCH)
CFLAGS   ?= -O2 -std=c11 -Wall -Wextra $(MARCH)

BUILD    := build
LIB      := $(BUILD)/libkakeya_abi.so
LIB_A    := $(BUILD)/libkakeya_abi.a
CLI      := $(BUILD)/ka
TESTBIN  := $(BUILD)/test_abi

# macOS 上动态库后缀是 .dylib，且共享库允许未定义符号。
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB      := $(BUILD)/libkakeya_abi.dylib
  SHARED   := -dynamiclib -undefined dynamic_lookup
else
  SHARED   := -shared
endif

.PHONY: all lib cli check clean

all: lib cli

lib: $(LIB)
$(LIB): abi/ka_abi.cpp abi/ka_abi.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SHARED) abi/ka_abi.cpp -o $(LIB)

# 静态库：给嵌入式 / 不想带动态库的场景。
$(LIB_A): abi/ka_abi.cpp abi/ka_abi.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c abi/ka_abi.cpp -o $(BUILD)/ka_abi.o
	ar rcs $(LIB_A) $(BUILD)/ka_abi.o

cli: $(CLI)
# CLI 静态链接本库，产出单文件可执行程序（无运行时依赖）。
$(CLI): abi/ka_cli.cpp abi/ka_abi.cpp abi/ka_abi.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Iabi abi/ka_cli.cpp abi/ka_abi.cpp -o $(CLI)

# 用纯 C 编写的测试程序验证 ABI 能被 C 调用（而非只能用 C++ 调用）。
$(TESTBIN): abi/test_abi.c abi/ka_abi.h $(LIB)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Iabi abi/test_abi.c -L$(BUILD) -lkakeya_abi -o $(TESTBIN)

check: $(TESTBIN)
	./$(TESTBIN)

clean:
	rm -rf $(BUILD)
