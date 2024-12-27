#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 宏定义，用于索引子矩阵
#define IDX(i, j, n) ((i)*(n) + (j))

// 初始化矩阵为随机数
void initialize_matrix(double *mat, int n) {
    for(int i = 0; i < n*n; i++) {
        mat[i] = rand() % 10; // 随机数 0-9
    }
}

// 打印矩阵（用于调试）
void print_matrix(double *mat, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%6.2f ", mat[IDX(i,j,n)]);
        }
        printf("\n");
    }
}

// 矩阵相乘加运算：C += A * B
void matmul_add(double *A, double *B, double *C, int n) {
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < n; k++) {
            for(int j = 0; j < n; j++) {
                C[IDX(i,j,n)] += A[IDX(i,k,n)] * B[IDX(k,j,n)];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 检查 p 是否为完全平方数
    int q = (int)sqrt((double)size);
    if(q * q != size) {
        if(rank == 0) {
            printf("进程数 p 必须是完全平方数。\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    // 固定矩阵大小 n x n
    int n = 4; // 固定为4
    // 检查 n 能否被 q 整除
    if(n % q != 0) {
        if(rank == 0) {
            printf("矩阵大小 n 必须能被 sqrt(p) 整除。\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    int block_size = n / q;
    
    // 创建二维网格通信器
    int dims[2] = {q, q};
    int periods[2] = {1, 1}; // 周期性，以便循环移动
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    
    // 获取当前进程在网格中的坐标
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    
    // 创建行通信器和列通信器，使用 MPI_Cart_sub
    MPI_Comm row_comm, col_comm;
    int remain_dims_row[2] = {0, 1}; // 保留列，变化行
    MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
    
    int remain_dims_col[2] = {1, 0}; // 保留行，变化列
    MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);
    
    // 分配内存给子矩阵
    double *A_block = (double*)malloc(block_size * block_size * sizeof(double));
    double *B_block = (double*)malloc(block_size * block_size * sizeof(double));
    double *C_block = (double*)malloc(block_size * block_size * sizeof(double));
    
    // 初始化 C_block 为 0
    for(int i = 0; i < block_size * block_size; i++) {
        C_block[i] = 0.0;
    }
    
    // 仅根进程初始化整个矩阵
    double *A = NULL;
    double *B = NULL;
    if(rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        initialize_matrix(A, n);
        initialize_matrix(B, n);
        printf("Matrix A:\n");
        print_matrix(A, n);
        printf("Matrix B:\n");
        print_matrix(B, n);
    }
    
    // 使用 MPI_Type_create_subarray 创建子矩阵数据类型
    MPI_Datatype submatrix_type;
    int sizes_array[2] = {n, n};
    int subsizes_array[2] = {block_size, block_size};
    int starts_array[2] = {0, 0};
    MPI_Type_create_subarray(2, sizes_array, subsizes_array, starts_array, MPI_ORDER_C, MPI_DOUBLE, &submatrix_type);
    
    // 创建一个调整过的类型，使其可以被 MPI_Scatterv 正确识别
    MPI_Datatype resized_submatrix_type;
    MPI_Type_create_resized(submatrix_type, 0, n * sizeof(double), &resized_submatrix_type);
    MPI_Type_commit(&resized_submatrix_type);
    
    // 定义发送计数和偏移量
    int *sendcounts = NULL;
    int *displs = NULL;
    if(rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            sendcounts[i] = 1;
            int row = i / q;
            int col = i % q;
            displs[i] = row * n + col;
        }
    }
    
    // 分发 A 和 B
    MPI_Scatterv(A, sendcounts, displs, resized_submatrix_type, A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, resized_submatrix_type, B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        free(sendcounts);
        free(displs);
    }
    
    // 释放全局矩阵
    if(rank == 0) {
        free(A);
        free(B);
    }
    
    // Fox 算法的主要循环
    for(int stage = 0; stage < q; stage++) {
        int root = (my_row + stage) % q;
        double *A_broadcast = (double*)malloc(block_size * block_size * sizeof(double));
        
        if(root == my_col) {
            // 当前处理器是广播的根，复制 A_block 到 A_broadcast
            for(int i = 0; i < block_size * block_size; i++) {
                A_broadcast[i] = A_block[i];
            }
        }
        
        // 广播 A_broadcast 到同一行的所有处理器
        MPI_Bcast(A_broadcast, block_size * block_size, MPI_DOUBLE, root, row_comm);
        
        // 执行 C_block += A_broadcast * B_block
        matmul_add(A_broadcast, B_block, C_block, block_size);
        
        free(A_broadcast);
        
        // 将 B_block 向上移动一位（循环移位）
        // 使用 MPI_Cart_shift 在列通信器内循环移位
        int src, dest;
        MPI_Cart_shift(col_comm, 0, -1, &src, &dest);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE, dest, 0, src, 0, col_comm, MPI_STATUS_IGNORE);
    }
    
    // 收集所有 C_block 到根进程
    double *C = NULL;
    if(rank == 0) {
        C = (double*)malloc(n * n * sizeof(double));
    }
    
    // 重新定义 sendcounts 和 displs 用于 Gatherv
    if(rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            sendcounts[i] = 1;
            int row = i / q;
            int col = i % q;
            displs[i] = row * n + col;
        }
    }
    
    MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE, C, sendcounts, displs, resized_submatrix_type, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        printf("Matrix C = A * B:\n");
        print_matrix(C, n);
        free(C);
        free(sendcounts);
        free(displs);
    }
    
    // 释放资源
    free(A_block);
    free(B_block);
    free(C_block);
    
    MPI_Type_free(&submatrix_type);
    MPI_Type_free(&resized_submatrix_type);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}
