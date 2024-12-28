#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef N
#define N 50
#endif

#define INDEX(i, j) (((i)*N)+(j))

// 初始化矩阵A，使其元素由索引i, j唯一确定
void initialize_matrix(double *a, int num_rows, int num_cols) {
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            a[i * num_cols + j] = (double)(i * num_cols + j); // 例如 A[i][j] = i * N + j
        }
    }
}

// 计算矩阵B的值
void comp(double *A, double *B, int a, int b) {
    for(int i = 1; i <= a; i++) {
        for(int j = 1; j <= b; j++) {
            B[INDEX(i, j)] = (A[INDEX(i-1, j)] + A[INDEX(i, j+1)] + A[INDEX(i+1, j)] + A[INDEX(i, j-1)]) / 4.0;
        }
    }
}

// 检查B与C是否相等
int check(double *B, double *C) {
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {
            if (fabs(B[INDEX(i, j)] - C[INDEX(i, j)]) >= 1e-2) {
                printf("B[%d,%d] = %lf not %lf!\n", i, j, B[INDEX(i, j)], C[INDEX(i, j)]);
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

    MPI_Datatype SubMat;
    int rows = sqrt(num_procs);
    int cols = num_procs / rows;
    int a = (N-2 + rows-1) / rows;
    int b = (N-2 + cols-1) / cols;
    int alloc_num = (a+2)*(b+2); // 每个进程需要额外的边界
    A = (double*)malloc(alloc_num*sizeof(double));
    B = (double*)malloc(alloc_num*sizeof(double));
    B2= (double*)malloc(alloc_num*sizeof(double));

    // Proc#0 初始化矩阵A
    if (id_procs == 0) {
        // 分配整个矩阵A用于初始化和验证
        double *full_A = (double*)malloc(N*N*sizeof(double));
        double *full_B2 = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        // 计算全局B2用于验证
        comp(full_A, full_B2, N-2, N-2);
        // 将初始化后的A拷贝到本地A
        for(int i = 0; i < a+2; i++) {
            for(int j = 0; j < b+2; j++) {
                A[INDEX(i, j)] = full_A[i * N + j];
            }
        }
        free(full_A);
        free(full_B2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 创建子矩阵的数据类型
    MPI_Type_vector(a, b, N, MPI_DOUBLE, &SubMat);
    MPI_Type_commit(&SubMat);

    if (id_procs == 0) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if (i == 0 && j == 0)
                    continue;
                MPI_Send(&A[INDEX(i*a, j*b)], 1, SubMat, j + cols*i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(A, 1, SubMat, 0, 0, MPI_COMM_WORLD, &status);
    }

    // 计算B
    comp(A, B, a, b);

    // 定义用于接收B的子矩阵的数据类型
    MPI_Datatype SubMat_B;
    MPI_Type_vector(a, b, a+2, MPI_DOUBLE, &SubMat_B);
    MPI_Type_commit(&SubMat_B);
    if (id_procs == 0) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if (i == 0 && j == 0)
                    continue;
                MPI_Recv(&B[INDEX(i*a+1, j*b+1)], 1, SubMat_B, i*cols+j, 1, MPI_COMM_WORLD, &status);
            }
        }
    } else {
        MPI_Send(&B[INDEX(1, 1)], 1, SubMat_B, 0, 1, MPI_COMM_WORLD);
    }

    // 主进程收集并打印结果
    if (id_procs == 0) {
        // 收集所有子矩阵到完整的矩阵B
        double *full_B = (double*)malloc(N*N*sizeof(double));
        // 初始化full_B为0
        for(int i = 0; i < N*N; i++) {
            full_B[i] = 0.0;
        }
        // 拷贝本地B到full_B
        for(int i = 1; i <= a; i++) {
            for(int j = 1; j <= b; j++) {
                full_B[i * N + j] = B[INDEX(i, j)];
            }
        }
        // 接收其他进程的B部分
        for(int p = 1; p < num_procs; p++) {
            int proc_row = p / cols;
            int proc_col = p % cols;
            for(int i = 1; i <= a; i++) {
                for(int j = 1; j <= b; j++) {
                    double value;
                    MPI_Recv(&value, 1, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, &status);
                    full_B[(proc_row * a + i) * N + (proc_col * b + j)] = value;
                }
            }
        }

        // 打印矩阵A和B
        // 首先初始化完整的矩阵A
        double *full_A = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        print_matrix(full_A, N, N, "A");
        print_matrix(full_B, N, N, "B");

        // 验证结果
        // 计算B2
        double *computed_B2 = (double*)malloc(N*N*sizeof(double));
        comp(full_A, computed_B2, N-2, N-2);
        if (check(full_B, computed_B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error!\n");
        }

        free(full_A);
        free(full_B);
        free(computed_B2);
    } else {
        // 发送B的部分到主进程
        for(int i = 1; i <= a; i++) {
            for(int j = 1; j <= b; j++) {
                MPI_Send(&B[INDEX(i, j)], 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
        }
    }

    free(A);
    free(B);
    free(B2);
    MPI_Type_free(&SubMat);
    MPI_Type_free(&SubMat_B);
    MPI_Finalize();
    return 0;
}
