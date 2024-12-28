#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 移除宏定义的 N
/*
#ifndef N
#define N 7
#endif
*/

// 将 N 定义为全局变量，默认值为 7
int N = 7;

// 定义 INDEX 为内联函数，因为 N 现在是一个变量
static inline int INDEX(int i, int j) {
    return i * N + j;
}

// 初始化矩阵 A，使其元素由索引 i, j 唯一确定
void random_array(double *a, int num_elements) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[INDEX(i, j)] = (double)(i * N + j);  
        }
    }

    // 仅在根进程打印矩阵 A
    printf("Matrix A (initialized with indices):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", a[INDEX(i, j)]);
        }
        printf("\n");
    }
}

// 初始化矩阵 B 为全 0
void initialize_matrix(double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[INDEX(i, j)] = 0.0;
        }
    }
}

// 计算矩阵 B 的值
void comp(double *A, double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int up = (i - 1 + N) % N;
            int down = (i + 1) % N;
            int left = (j - 1 + N) % N;
            int right = (j + 1) % N;

            B[INDEX(i, j)] = (A[INDEX(up, j)] + A[INDEX(i, right)] + A[INDEX(down, j)] + A[INDEX(i, left)]) / 4.0;
        }
    }
}

// 检查 B 与 C 是否相等
int check(double *B, double *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(B[INDEX(i, j)] - C[INDEX(i, j)]) >= 1e-2) {
                printf("B[%d,%d] = %lf not %lf!\n", i, j, B[INDEX(i, j)], C[INDEX(i, j)]);
                return 0;
            }
        }
    }
    return 1;
}

// 打印矩阵
void print_matrix(double *mat) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", mat[INDEX(i, j)]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    double *A, *B, *B2;

    int id_procs, num_procs, num_1;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);

    num_1 = num_procs - 1;

    // 根进程解析命令行参数以设置 N
    if (id_procs == 0) {
        if (argc > 1) {
            int input_N = atoi(argv[1]);
            if (input_N > 0) {
                N = input_N;
                printf("Matrix size N set to %d\n", N);
            } else {
                printf("Invalid N value provided. Using default N = 7.\n");
            }
        } else {
            printf("Using default matrix size N = %d.\n", N);
        }
    }

    // 广播 N 给所有进程
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 根据 N 分配内存
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    B2 = (double*)malloc(N * N * sizeof(double));

    // 初始化矩阵
    if (id_procs == num_1) {
        random_array(A, N * N);
        initialize_matrix(B);
        comp(A, B2);
    }

    // 所有进程在开始计算前同步
    MPI_Barrier(MPI_COMM_WORLD);

    // 开始计时
    double computation_start = MPI_Wtime();

    int ctn = 0;
    for (int i = 0; i < N-2; i++) {
        if (id_procs == num_1) {
            int dest = i % num_1;
            int tag = i / num_1;
            // 发送第 i 行的数据到目标进程
            MPI_Send(&A[INDEX(i, 0)], N, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }
    }

    // 接收数据
    for (int i = 0; i < (N-2)/num_1; i++) {
        if (id_procs != num_1) {
            MPI_Recv(&A[INDEX(ctn * 3, 0)], 3*N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
            ctn++;
        }
    }
    if (id_procs < (N-2) % num_1) {
        if (id_procs != num_1) { // 进程 num_1 不接收
            MPI_Recv(&A[INDEX(ctn * 3, 0)], 3*N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
            ctn++;
        }
    }

    // 计算 B
    if (id_procs != num_1) {
        for (int i = 1; i <= ctn; i++) {
            for (int j = 1; j < N-1; j++) {
                B[INDEX(i, j)] = (A[INDEX(i-1, j)] + A[INDEX(i, j+1)] + A[INDEX(i+1, j)] + A[INDEX(i, j-1)]) / 4.0;
            }
        }
    }

    // 发送计算结果回根进程
    for (int i = 0; i < N-2; i++) {
        if (id_procs == num_1) {
            int src = i % num_1;
            MPI_Recv(&B[INDEX(i+1, 1)], N-2, MPI_DOUBLE, src, i / num_1 + N, MPI_COMM_WORLD, &status);
        } else {
            for (int j = 0; j < ctn; j++) {
                MPI_Send(&B[INDEX(j+1, 1)], N-2, MPI_DOUBLE, num_1, j + N, MPI_COMM_WORLD);
            }
        }
    }

    // 停止计时
    double computation_end = MPI_Wtime();
    double local_elapsed = computation_end - computation_start;

    // 汇总所有进程中的最大耗时
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 根进程处理输出
    if (id_procs == num_1) {
        // 验证结果
        if (check(B, B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error Occurred!\n");
        }

        // 打印矩阵 A 和 B
        printf("\nMatrix A (initialized with indices):\n");
        print_matrix(A);

        printf("\nMatrix B (calculated result):\n");
        print_matrix(B);

        // 打印计算用时
        printf("\nComputation Time: %lf seconds\n", max_elapsed);
    }

    // 释放内存
    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
