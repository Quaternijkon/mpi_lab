#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 矩阵乘法函数：C += A * B
void matrixMultiply(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 确保进程数为平方数
    int grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            printf("进程数必须是一个完全平方数！\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int n = 4; // 矩阵维度（应可被 grid_size 整除）
    int block_size = n / grid_size; // 子块的大小

    // 创建 2D 进程网格
    MPI_Comm grid_comm;
    int dims[2] = {grid_size, grid_size};
    int periods[2] = {1, 1}; // 允许环绕通信
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int row = coords[0];
    int col = coords[1];

    // 初始化矩阵（简单初始化为 rank 值）
    int* A = (int*)malloc(block_size * block_size * sizeof(int));
    int* B = (int*)malloc(block_size * block_size * sizeof(int));
    int* C = (int*)malloc(block_size * block_size * sizeof(int));
    int* A_temp = (int*)malloc(block_size * block_size * sizeof(int));
    int* B_temp = (int*)malloc(block_size * block_size * sizeof(int));

    for (int i = 0; i < block_size * block_size; ++i) {
        A[i] = rank; // 示例数据
        B[i] = rank; // 示例数据
        C[i] = 0;
    }

    for (int k = 0; k < grid_size; ++k) {
        // 广播 A 子块
        int Bcaster = row * grid_size + (row + k) % grid_size;
        if (rank == Bcaster) {
            for (int i = 0; i < block_size * block_size; ++i) {
                A_temp[i] = A[i];
            }
            for (int l = 0; l < grid_size; ++l) {
                int dst = row * grid_size + l;
                if (dst != rank) {
                    MPI_Send(A_temp, block_size * block_size, MPI_INT, dst, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(A_temp, block_size * block_size, MPI_INT, Bcaster, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 执行矩阵乘法
        matrixMultiply(A_temp, B, C, block_size);

        // 循环移动 B 子块
        int send_B = (row * grid_size + col + grid_size - 1) % grid_size;
        int recv_B = (row * grid_size + col + 1) % grid_size;

        if ((row % 2) == 0) {
            MPI_Send(B, block_size * block_size, MPI_INT, send_B, 1, MPI_COMM_WORLD);
            MPI_Recv(B_temp, block_size * block_size, MPI_INT, recv_B, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(B_temp, block_size * block_size, MPI_INT, recv_B, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(B, block_size * block_size, MPI_INT, send_B, 1, MPI_COMM_WORLD);
        }

        // 更新 B 矩阵
        for (int i = 0; i < block_size * block_size; ++i) {
            B[i] = B_temp[i];
        }
    }

    // 输出结果
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            printf("进程 %d 结果:\n", rank);
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    printf("%d ", C[j * block_size + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(A_temp);
    free(B_temp);

    MPI_Finalize();
    return 0;
}
