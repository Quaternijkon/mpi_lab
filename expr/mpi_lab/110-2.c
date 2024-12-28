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
    double *A, *B, *B2;

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

    A = (double*)malloc(local_rows * local_cols * sizeof(double));
    B = (double*)malloc(local_rows * local_cols * sizeof(double));
    B2= (double*)malloc(local_rows * local_cols * sizeof(double));

    // 初始化所有子矩阵A为0
    for(int i = 0; i < local_rows * local_cols; i++) {
        A[i] = 0.0;
        B[i] = 0.0;
        B2[i] = 0.0;
    }

    // Proc#0 初始化全局矩阵A并分发子矩阵
    if (id_procs == 0) {
        double *full_A = (double*)malloc(N*N*sizeof(double));
        double *full_B2 = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        // 计算全局B2用于验证
        // 这里为了简化，直接调用 comp 函数在全局矩阵上计算
        // 需要定义一个临时矩阵用于全局计算
        double *temp_A = (double*)malloc(N*N*sizeof(double));
        double *temp_B = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(temp_A, N, N);
        // 重新实现 comp 函数对全局矩阵进行计算
        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                temp_B[GLOBAL_INDEX(i,j)] = (temp_A[GLOBAL_INDEX(i-1,j)] +
                                             temp_A[GLOBAL_INDEX(i,j+1)] +
                                             temp_A[GLOBAL_INDEX(i+1,j)] +
                                             temp_A[GLOBAL_INDEX(i,j-1)]) / 4.0;
            }
        }
        // 拷贝到 full_B2
        for(int i = 0; i < N*N; i++) {
            full_B2[i] = temp_B[i];
        }
        free(temp_A);
        free(temp_B);
        // 将全局A分发到各个进程
        for(int p = 0; p < num_procs; p++) {
            int proc_row = p / cols;
            int proc_col = p % cols;
            // 定义发送缓冲区
            double *send_buffer = (double*)malloc(local_rows * local_cols * sizeof(double));
            // 填充send_buffer，包括边界
            for(int i = 0; i < local_rows; i++) {
                for(int j = 0; j < local_cols; j++) {
                    // 正确计算全局索引，考虑内部偏移
                    int global_i = 1 + proc_row * a + (i - 1);
                    int global_j = 1 + proc_col * b + (j - 1);
                    if(global_i < 1 || global_i >= N-1 || global_j < 1 || global_j >= N-1)
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
        free(full_A);
        free(full_B2);
    }
    else {
        // 接收子矩阵A
        MPI_Recv(A, local_rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // **添加计时器开始**
    double start_time = MPI_Wtime();

    // 计算B
    comp(A, B, a, b, local_cols);

    // 确保边界元素保持为0
    // 计算当前进程在进程网格中的行和列
    int proc_row = id_procs / cols;
    int proc_col = id_procs % cols;
    // 计算全局起始行和列
    int start_i = proc_row * a;
    int start_j = proc_col * b;
    // 遍历本地B的内部元素
    for(int i = 1; i <= a; i++) {
        for(int j = 1; j <= b; j++) {
            // 正确计算全局索引，考虑内部偏移
            int global_i = 1 + proc_row * a + (i - 1);
            int global_j = 1 + proc_col * b + (j - 1);
            // 如果是全局边界，则设置为0
            if(global_i == 0 || global_j == 0 || global_i == N-1 || global_j == N-1) {
                B[LOCAL_INDEX(i, j, local_cols)] = 0.0;
            }
        }
    }

    // **添加计时器结束**
    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed;

    // 计算所有进程中的最大用时
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all B parts to Proc#0
    if (id_procs == 0) {
        double *full_B = (double*)malloc(N*N*sizeof(double));
        // 初始化全局B为0
        for(int i = 0; i < N*N; i++) {
            full_B[i] = 0.0;
        }
        // 收集自身的B部分
        for(int i = 1; i <= a; i++) {
            for(int j = 1; j <= b; j++) {
                int global_i = 1 + 0 * a + (i - 1);
                int global_j = 1 + 0 * b + (j - 1);
                if(global_i < N && global_j < N)
                    full_B[GLOBAL_INDEX(global_i, global_j)] = B[LOCAL_INDEX(i, j, local_cols)];
            }
        }
        // 接收其他进程的B部分
        for(int p = 1; p < num_procs; p++) {
            double *recv_buffer = (double*)malloc(local_rows * local_cols * sizeof(double));
            MPI_Recv(recv_buffer, local_rows * local_cols, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            int proc_row_p = p / cols;
            int proc_col_p = p % cols;
            for(int i = 1; i <= a; i++) {
                for(int j = 1; j <= b; j++) {
                    int global_i = 1 + proc_row_p * a + (i - 1);
                    int global_j = 1 + proc_col_p * b + (j - 1);
                    if(global_i < N && global_j < N)
                        full_B[GLOBAL_INDEX(global_i, global_j)] = recv_buffer[LOCAL_INDEX(i, j, local_cols)];
                }
            }
            free(recv_buffer);
        }

        // 打印矩阵A和B
        double *full_A = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        print_matrix(full_A, N, N, "A");
        print_matrix(full_B, N, N, "B");

        // 打印计算用时
        printf("Computation Time: %lf seconds\n", max_elapsed);

        // 验证结果
        // 计算全局B2
        double *computed_B2 = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        // 重新计算 computed_B2
        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                computed_B2[GLOBAL_INDEX(i,j)] = (full_A[GLOBAL_INDEX(i-1,j)] +
                                                 full_A[GLOBAL_INDEX(i,j+1)] +
                                                 full_A[GLOBAL_INDEX(i+1,j)] +
                                                 full_A[GLOBAL_INDEX(i,j-1)]) / 4.0;
            }
        }
        if (check(full_B, computed_B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error!\n");
        }

        free(full_A);
        free(full_B);
        free(computed_B2);
    }
    else {
        // 发送B的部分到Proc#0
        MPI_Send(B, local_rows * local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    // 只有根进程需要打印计时信息，因此其他进程不需要打印
    // 计时信息已经通过MPI_Reduce汇总并在根进程打印

    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
