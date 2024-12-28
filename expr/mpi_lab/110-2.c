#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef N
#define N 7 // 根据您的输出，假设N=7
#endif

// 本地索引宏
#define LOCAL_INDEX(i, j, cols) (((i)*(cols)) + (j))

// 全局索引宏，用于打印完整矩阵
#define GLOBAL_INDEX(i, j) (((i)*N)+(j))

// 初始化矩阵A，使其元素由索引i, j唯一确定
void initialize_matrix(double *a, int num_rows, int num_cols) {
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            a[i * num_cols + j] = (double)(i * num_cols + j); // 例如 A[i][j] = i * N + j
        }
    }
}

// 计算矩阵B的值
void comp(double *A, double *B, int a, int b, int local_cols) {
    for(int i = 1; i <= a; i++) {
        for(int j = 1; j <= b; j++) {
            B[LOCAL_INDEX(i, j, local_cols)] = (A[LOCAL_INDEX(i-1, j, local_cols)] +
                                               A[LOCAL_INDEX(i, j+1, local_cols)] +
                                               A[LOCAL_INDEX(i+1, j, local_cols)] +
                                               A[LOCAL_INDEX(i, j-1, local_cols)]) / 4.0;
        }
    }
}

// 检查B与C是否相等
int check(double *B, double *C) {
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {
            if (fabs(B[GLOBAL_INDEX(i, j)] - C[GLOBAL_INDEX(i, j)]) >= 1e-2) {
                printf("B[%d,%d] = %lf not %lf!\n", i, j, B[GLOBAL_INDEX(i, j)], C[GLOBAL_INDEX(i, j)]);
                return 0;
            }
        }
    }
    return 1;
}

// 打印矩阵
void print_matrix(double *matrix, int num_rows, int num_cols, const char *name) {
    printf("Matrix %s:\n", name);
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            printf("%6.2lf ", matrix[i * num_cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    double *A, *B;
    double *B2_full = NULL; // 仅在主进程中使用

    int id_procs, num_procs;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);

    // 假设num_procs是一个完全平方数
    int rows = (int)sqrt(num_procs);
    int cols = num_procs / rows;
    if(rows * cols != num_procs) {
        if(id_procs == 0)
            printf("Number of processes must be a perfect square.\n");
        MPI_Finalize();
        return 0;
    }

    int a = (N-2 + rows-1) / rows; // 每行分配的子矩阵行数
    int b = (N-2 + cols-1) / cols; // 每列分配的子矩阵列数
    int local_rows = a + 2; // 包含上下边界
    int local_cols = b + 2; // 包含左右边界

    // 根据进程号分配B2的大小
    if(id_procs == 0){
        // 主进程分配全局B2
        A = (double*)malloc(local_rows * local_cols * sizeof(double));
        B = (double*)malloc(local_rows * local_cols * sizeof(double));
        B2_full = (double*)malloc(N * N * sizeof(double));
        if (A == NULL || B == NULL || B2_full == NULL) {
            fprintf(stderr, "Memory allocation failed for main process.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    else{
        // 其他进程分配局部B2
        A = (double*)malloc(local_rows * local_cols * sizeof(double));
        B = (double*)malloc(local_rows * local_cols * sizeof(double));
        // 其他进程不需要使用B2，但为了保持代码一致性，可以分配
        // 或者将B2去除
        if (A == NULL || B == NULL) {
            fprintf(stderr, "Memory allocation failed for process %d.\n", id_procs);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // 初始化所有子矩阵A和B为0
    for(int i = 0; i < local_rows * local_cols; i++) {
        A[i] = 0.0;
        B[i] = 0.0;
    }

    // 主进程初始化全局矩阵A并分发子矩阵
    if (id_procs == 0) {
        double *full_A = (double*)malloc(N * N * sizeof(double));
        double *full_B2 = (double*)malloc(N * N * sizeof(double));
        if (full_A == NULL || full_B2 == NULL) {
            fprintf(stderr, "Memory allocation failed for full_A or full_B2 in main process.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        initialize_matrix(full_A, N, N);

        // 计算全局B2用于验证
        initialize_matrix(full_A, N, N); // 再次初始化temp_A
        for(int i = 0; i < N * N; i++) {
            full_B2[i] = 0.0; // Initialize full_B2 to 0
        }

        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                full_B2[GLOBAL_INDEX(i, j)] = (full_A[GLOBAL_INDEX(i-1, j)] +
                                               full_A[GLOBAL_INDEX(i, j+1)] +
                                               full_A[GLOBAL_INDEX(i+1, j)] +
                                               full_A[GLOBAL_INDEX(i, j-1)]) / 4.0;
            }
        }

        // 分发全局A到各个进程
        for(int p = 0; p < num_procs; p++) {
            int proc_row = p / cols;
            int proc_col = p % cols;
            double *send_buffer = (double*)malloc(local_rows * local_cols * sizeof(double));
            if (send_buffer == NULL) {
                fprintf(stderr, "Memory allocation failed for send_buffer in main process.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            // 填充send_buffer，包括边界
            for(int i = 0; i < local_rows; i++) {
                for(int j = 0; j < local_cols; j++) {
                    int global_i = proc_row * a + (i - 1);
                    int global_j = proc_col * b + (j - 1);
                    if(global_i < 0 || global_i >= N || global_j < 0 || global_j >= N)
                        send_buffer[i * local_cols + j] = 0.0; // 边界填充为0
                    else
                        send_buffer[i * local_cols + j] = full_A[GLOBAL_INDEX(global_i, global_j)];
                }
            }

            if(p == 0) {
                // 直接拷贝到本地A
                for(int i = 0; i < local_rows * local_cols; i++) {
                    A[i] = send_buffer[i];
                }
            }
            else {
                // 发送给其他进程
                MPI_Send(send_buffer, local_rows * local_cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }

            free(send_buffer);
        }

        // 将全局B2复制到主进程的B2_full
        for(int i = 0; i < N * N; i++) {
            B2_full[i] = full_B2[i];
        }

        free(full_A);
        free(full_B2);
    }
    else {
        // 其他进程接收子矩阵A
        MPI_Recv(A, local_rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 计算B
    comp(A, B, a, b, local_cols);

    // 收集所有进程的B部分到主进程
    if (id_procs == 0) {
        double *full_B = (double*)malloc(N * N * sizeof(double));
        if (full_B == NULL) {
            fprintf(stderr, "Memory allocation failed for full_B in main process.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // 初始化全局B为0
        for(int i = 0; i < N * N; i++) {
            full_B[i] = 0.0;
        }

        // 收集主进程自身的B部分
        for(int i = 1; i <= a; i++) {
            for(int j = 1; j <= b; j++) {
                int global_i = 0 * a + (i - 1);
                int global_j = 0 * b + (j - 1);
                if(global_i < N && global_j < N)
                    full_B[GLOBAL_INDEX(global_i, global_j)] = B[LOCAL_INDEX(i, j, local_cols)];
            }
        }

        // 接收其他进程的B部分
        for(int p = 1; p < num_procs; p++) {
            double *recv_buffer = (double*)malloc(local_rows * local_cols * sizeof(double));
            if (recv_buffer == NULL) {
                fprintf(stderr, "Memory allocation failed for recv_buffer in main process.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            MPI_Recv(recv_buffer, local_rows * local_cols, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            int proc_row = p / cols;
            int proc_col = p % cols;

            for(int i = 1; i <= a; i++) {
                for(int j = 1; j <= b; j++) {
                    int global_i = proc_row * a + (i - 1);
                    int global_j = proc_col * b + (j - 1);
                    if(global_i < N && global_j < N)
                        full_B[GLOBAL_INDEX(global_i, global_j)] = recv_buffer[LOCAL_INDEX(i, j, local_cols)];
                }
            }

            free(recv_buffer);
        }

        // **确保边界元素保持为0**
        // 设置第一行和最后一行为0
        for(int j = 0; j < N; j++) {
            full_B[GLOBAL_INDEX(0, j)] = 0.0;
            full_B[GLOBAL_INDEX(N-1, j)] = 0.0;
        }
        // 设置第一列和最后一列为0
        for(int i = 0; i < N; i++) {
            full_B[GLOBAL_INDEX(i, 0)] = 0.0;
            full_B[GLOBAL_INDEX(i, N-1)] = 0.0;
        }

        // 打印矩阵A和B
        double *full_A = (double*)malloc(N * N * sizeof(double));
        if(full_A == NULL){
            fprintf(stderr, "Memory allocation failed for full_A in main process.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        initialize_matrix(full_A, N, N);
        print_matrix(full_A, N, N, "A");
        print_matrix(full_B, N, N, "B");

        // 验证结果
        if (check(full_B, B2_full)) {
            printf("Done. No Error\n");
        } else {
            printf("Error!\n");
        }

        free(full_A);
        free(full_B);
        free(full_B2);
        free(B2_full);
    }
    else {
        // 其他进程发送B的部分到主进程
        MPI_Send(B, local_rows * local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    // 统一释放内存
    free(A);
    free(B);
    if(id_procs == 0){
        // 主进程已经释放了 B2_full 和 full_B2
    }
    else{
        free(B2); // 其他进程释放 B2
    }

    MPI_Finalize();
    return 0;
}
