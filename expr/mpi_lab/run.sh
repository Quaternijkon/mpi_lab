#!/bin/bash

# 设置进程数量
NUM_PROCESSES=32

# 运行 MPI 程序
mpirun -np $NUM_PROCESSES ./lab-a-2
