#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 根据特定规则初始化矩阵 A 和 B
void initialize_matrices(double *A, double *B, int n, int block_size, int rank, int size, int coords[2]) {
    // 计算全局起始行和列索引
    int sqrt_p = (int)sqrt((double)size);
    int row_offset = coords[0] * block_size;
    int col_offset = coords[1] * block_size;
    
    for(int i = 0; i < block_size; i++) {
        for(int j = 0; j < block_size; j++) {
            int global_i = row_offset + i;
            int global_j = col_offset + j;
            A[i * block_size + j] = global_i + global_j;      // 规则：A[i][j] = i + j
            B[i * block_size + j] = global_i - global_j;      // 规则：B[i][j] = i - j
        }
    }
}

// 打印矩阵（仅在根进程调用）
void print_matrix(double *matrix, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%lf ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

// 矩阵乘法并累加结果到 C
void multiply_add(double *A, double *B, double *C, int block_size) {
    for(int i = 0; i < block_size; i++) {
        for(int j = 0; j < block_size; j++) {
            for(int k = 0; k < block_size; k++) {
                C[i * block_size + j] += A[i * block_size + k] * B[k * block_size + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 4; // 矩阵大小，假设为 8x8，可根据需要调整
    int sqrt_p;
    int block_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 确保进程数为完全平方数
    sqrt_p = (int)sqrt((double)size);
    if(sqrt_p * sqrt_p != size) {
        if(rank == 0) {
            printf("进程数必须为完全平方数。\n");
        }
        MPI_Finalize();
        return -1;
    }
    
    // 确保矩阵大小能被 sqrt_p 整除
    if(n % sqrt_p != 0) {
        if(rank == 0) {
            printf("矩阵大小必须能被 sqrt(p) 整除。\n");
        }
        MPI_Finalize();
        return -1;
    }
    
    block_size = n / sqrt_p;
    
    // 创建二维网格通信子
    MPI_Comm grid_comm;
    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {1, 1}; // 环形通信
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    
    // 创建行和列通信子
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);
    
    // 分配内存给子矩阵
    double *A_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *B_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *C_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *A_temp = (double *)malloc(block_size * block_size * sizeof(double));
    
    // 初始化 C 为 0
    for(int i = 0; i < block_size * block_size; i++) {
        C_block[i] = 0.0;
    }
    
    // 初始化矩阵 A 和 B
    initialize_matrices(A_block, B_block, n, block_size, rank, size, coords);
    
    // 创建临时缓冲区用于广播
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int step = 0; step < sqrt_p; step++) {
        int root = (my_row + step) % sqrt_p;
        
        if(root == my_col) {
            // 发送 A_block 到广播
            MPI_Bcast(A_block, block_size * block_size, MPI_DOUBLE, root, row_comm);
            // 将 A_block 复制到 A_temp
            for(int i = 0; i < block_size * block_size; i++) {
                A_temp[i] = A_block[i];
            }
        } else {
            // 接收广播的 A_block
            MPI_Bcast(A_temp, block_size * block_size, MPI_DOUBLE, root, row_comm);
        }
        
        // 进行乘法累加
        multiply_add(A_temp, B_block, C_block, block_size);
        
        // 循环上移 B_block
        int source, dest;
        MPI_Cart_shift(grid_comm, 1, -1, &source, &dest);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE, dest, 0, source, 0, grid_comm, MPI_STATUS_IGNORE);
    }
    
    // 收集所有 C_block 到根进程
    double *C = NULL;
    if(rank == 0) {
        C = (double *)malloc(n * n * sizeof(double));
    }
    
    // 使用 MPI_Gather 收集所有 C_block
    MPI_Gather(C_block, block_size * block_size, MPI_DOUBLE, C, block_size * block_size, MPI_DOUBLE, 0, grid_comm);
    
    // 在根进程打印结果
    if(rank == 0) {
        printf("结果矩阵 C:\n");
        print_matrix(C, n);
        free(C);
    }
    
    // 释放资源
    free(A_block);
    free(B_block);
    free(C_block);
    free(A_temp);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}
