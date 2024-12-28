#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int N = 7;

#define INDEX(i, j) (((i) * N) + (j))

void random_array(double *a, int num) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[INDEX(i, j)] = i * N + j;  
        }
    }

    printf("Matrix A (initialized with indices):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", a[INDEX(i, j)]);
        }
        printf("\n");
    }
}

void initialize_matrix(double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[INDEX(i, j)] = 0.0;
        }
    }
}

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

    // 主进程解析命令行参数并设置 N
    if (id_procs == num_1) {
        if (argc >= 2) {
            int input_N = atoi(argv[1]);
            if (input_N > 0) {
                N = input_N;
            } else {
                printf("Invalid matrix size provided. Using default N = 7.\n");
            }
        }
    }

    // 广播 N 给所有进程
    MPI_Bcast(&N, 1, MPI_INT, num_1, MPI_COMM_WORLD);

    // 所有进程根据 N 分配内存
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    B2 = (double*)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || B2 == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed for N=%d\n", id_procs, N);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (id_procs == num_1) {
        random_array(A, N * N);
        initialize_matrix(B);
        comp(A, B2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 开始计时
    double start_time, end_time, elapsed_time, max_time;
    start_time = MPI_Wtime();

    int ctn = 0;
    for (int i = 0; i < N - 2; i++) {
        if (id_procs == num_1) {
            int dest = i % num_1;
            int tag = i / num_1;
            MPI_Send(&A[INDEX(i, 0)], N * 3, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < (N - 2) / num_1; i++) {
        if (id_procs != num_1) {
            MPI_Recv(&A[INDEX(3 * ctn, 0)], 3 * N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
            ctn++;
        }
    }
    if (id_procs < (N - 2) % num_1) {
        if (id_procs != num_1) { // 确保只有非主进程接收
            MPI_Recv(&A[INDEX(ctn * 3, 0)], 3 * N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
            ctn++;
        }
    }

    if (id_procs != num_1) {
        for (int i = 1; i <= ctn; i++) {
            for (int j = 1; j < N - 1; j++) {
                // 确保索引不越界
                if ((i - 1) >= 0 && (i + 1) < N && (j - 1) >= 0 && (j + 1) < N) {
                    B[INDEX(i, j)] = (A[INDEX(i - 1, j)] + A[INDEX(i, j + 1)] + A[INDEX(i + 1, j)] + A[INDEX(i, j - 1)]) / 4.0;
                }
            }
        }
    }

    for (int i = 0; i < N - 2; i++) {
        if (id_procs == num_1) {
            int src = i % num_1;
            MPI_Recv(&B[INDEX(i + 1, 1)], N - 2, MPI_DOUBLE, src, i / num_1 + N, MPI_COMM_WORLD, &status);
        } else {
            for (int j = 0; j < ctn; j++) {
                MPI_Send(&B[INDEX(j + 1, 1)], N - 2, MPI_DOUBLE, num_1, j + N, MPI_COMM_WORLD);
            }
        }
    }

    // 停止计时
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // 计算所有进程的最大用时
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, num_1, MPI_COMM_WORLD);

    if (id_procs == num_1) {
        // 如果需要检查结果，可以取消注释以下代码
        /*
        if (check(B, B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error Occurred!\n");
        }
        */

        printf("\nMatrix A (initialized with indices):\n");
        print_matrix(A);

        printf("\nMatrix B (calculated result):\n");
        print_matrix(B);

        printf("\nTime elapsed for computing matrix B: %lf seconds\n", max_time);
    }

    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
