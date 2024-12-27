#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4  // 矩阵大小 N x N
#define ROOT 0

void matmul(float *A, float *B, float *C, int size, int rank, int p) {
    // Step 1: 分配局部矩阵
    int sqrt_p = (int) sqrt(p);
    int block_size = size / sqrt_p;
    float *local_A = (float *)malloc(block_size * size * sizeof(float));
    float *local_B = (float *)malloc(block_size * size * sizeof(float));
    float *local_C = (float *)malloc(block_size * size * sizeof(float));

    // 初始化矩阵A和B的局部块
    MPI_Scatter(A, block_size * size, MPI_FLOAT, local_A, block_size * size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(B, block_size * size, MPI_FLOAT, local_B, block_size * size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    // Step 2: 执行FOX算法
    for (int step = 0; step < sqrt_p; step++) {
        // 1. 每个进程执行局部矩阵乘法
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_C[i * block_size + j] = 0;
                for (int k = 0; k < size; k++) {
                    local_C[i * block_size + j] += local_A[i * size + k] * local_B[k * size + j];
                }
            }
        }

        // 2. 将计算结果发到相应的进程
        int row_shift = (rank / sqrt_p + step) % sqrt_p;
        int col_shift = (rank % sqrt_p + step) % sqrt_p;
        
        // 交换矩阵的行/列数据
        MPI_Sendrecv(local_A, block_size * size, MPI_FLOAT, 
                     (rank + row_shift) % sqrt_p, 0, 
                     local_A, block_size * size, MPI_FLOAT, 
                     (rank + row_shift) % sqrt_p, 0, MPI_COMM_WORLD);
        
        MPI_Sendrecv(local_B, block_size * size, MPI_FLOAT,
                     (rank + col_shift) % sqrt_p, 0,
                     local_B, block_size * size, MPI_FLOAT,
                     (rank + col_shift) % sqrt_p, 0, MPI_COMM_WORLD);
    }
    
    // Step 3: 汇总计算结果
    MPI_Gather(local_C, block_size * size, MPI_FLOAT, C, block_size * size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    free(local_A);
    free(local_B);
    free(local_C);
}

int main(int argc, char *argv[]) {
    int rank, size;
    float *A, *B, *C;
    
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == ROOT) {
            printf("This program requires 4 MPI processes.\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    // 为A、B和C矩阵分配空间
    if (rank == ROOT) {
        A = (float *)malloc(N * N * sizeof(float));
        B = (float *)malloc(N * N * sizeof(float));
        C = (float *)malloc(N * N * sizeof(float));
        
        // 初始化矩阵A和B
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
            }
        }
    }

    // 调用矩阵乘法函数
    matmul(A, B, C, N, rank, size);

    // 输出结果
    if (rank == ROOT) {
        printf("Matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", C[i * N + j]);
            }
            printf("\n");
        }

        // 释放内存
        free(A);
        free(B);
        free(C);
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}
