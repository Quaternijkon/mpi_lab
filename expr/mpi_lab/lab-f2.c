#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define TAG_UP 0
#define TAG_DOWN 1
#define TAG_LEFT 2
#define TAG_RIGHT 3


int row_block_partition(int N, int P, int rank, double ***A_local, double ***B_local, int *local_N, int *start_row);
int checkerboard_partition(int N, int P, int rank, double ***A_local, double ***B_local, int *local_rows, int *local_cols, int *start_row, int *start_col, MPI_Comm *grid_comm);
void gather_and_print_row(int N, int P, int rank, double **A_local, double **B_local, int local_N);
void gather_and_print_checkerboard(int N, int P, int rank, double **A_local, double **B_local, int local_rows, int local_cols, MPI_Comm grid_comm);
void initialize_A_local_row(int local_N, int N, int start_row, double **A_local);
void initialize_A_local_checkerboard(int local_rows, int local_cols, int start_row, int start_col, double **A_local, int N);
void free_memory_row(double **A_local, double **B_local, int local_N);
void free_memory_checkerboard(double **A_local, double **B_local, int local_rows);

int main(int argc, char *argv[]) {
    int rank, P;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc != 3) {
        if(rank == 0) {
            printf("Usage: %s <partition_type> <N>\n", argv[0]);
            printf("partition_type: row | checkerboard\n");
            printf("N: matrix size (NxN)\n");
        }
        MPI_Finalize();
        return -1;
    }

    char partition_type[20];
    strncpy(partition_type, argv[1], 19);
    partition_type[19] = '\0';
    int N = atoi(argv[2]);

    if(rank == 0) {
        printf("Matrix size: %d x %d\n", N, N);
        printf("Number of processes: %d\n", P);
        printf("Partition type: %s\n", partition_type);
    }

    if(strcmp(partition_type, "row") == 0) {
        
        double **A_local, **B_local;
        int local_N, start_row;
        if(row_block_partition(N, P, rank, &A_local, &B_local, &local_N, &start_row) != 0) {
            MPI_Finalize();
            return -1;
        }

        
        for(int i = 1; i <= local_N; i++) {
            for(int j = 1; j < N-1; j++) {
                B_local[i][j] = (A_local[i-1][j] + A_local[i][j+1] + A_local[i+1][j] + A_local[i][j-1]) / 4.0;
            }
        }

        
        gather_and_print_row(N, P, rank, A_local, B_local, local_N);

        
        free_memory_row(A_local, B_local, local_N);
    }
    else if(strcmp(partition_type, "checkerboard") == 0) {
        
        double **A_local, **B_local;
        int local_rows, local_cols, start_row, start_col;
        MPI_Comm grid_comm;
        if(checkerboard_partition(N, P, rank, &A_local, &B_local, &local_rows, &local_cols, &start_row, &start_col, &grid_comm) != 0) {
            MPI_Finalize();
            return -1;
        }

        
        for(int i = 1; i <= local_rows; i++) {
            for(int j = 1; j <= local_cols; j++) {
                B_local[i][j] = (A_local[i-1][j] + A_local[i][j+1] + A_local[i+1][j] + A_local[i][j-1]) / 4.0;
            }
        }

        
        gather_and_print_checkerboard(N, P, rank, A_local, B_local, local_rows, local_cols, grid_comm);

        
        free_memory_checkerboard(A_local, B_local, local_rows);
        MPI_Comm_free(&grid_comm);
    }
    else {
        if(rank == 0) {
            printf("Unknown partition type: %s. Use 'row' or 'checkerboard'.\n", partition_type);
        }
        MPI_Finalize();
        return -1;
    }

    MPI_Finalize();
    return 0;
}

/**
 * 按行块连续划分的实现
 * 返回 0 表示成功，-1 表示失败
 */
int row_block_partition(int N, int P, int rank, double ***A_local_ptr, double ***B_local_ptr, int *local_N_ptr, int *start_row_ptr) {
    
    int rows_per_proc = N / P;
    int remainder = N % P;
    int local_N = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    
    double **A_local = (double **)malloc((local_N + 2) * sizeof(double *));
    double **B_local = (double **)malloc((local_N + 2) * sizeof(double *));
    if(A_local == NULL || B_local == NULL) {
        printf("Process %d: Memory allocation failed.\n", rank);
        return -1;
    }
    for(int i = 0; i < local_N + 2; i++) {
        A_local[i] = (double *)malloc(N * sizeof(double));
        B_local[i] = (double *)malloc(N * sizeof(double));
        if(A_local[i] == NULL || B_local[i] == NULL) {
            printf("Process %d: Memory allocation failed.\n", rank);
            return -1;
        }
    }

    
    
    for(int i = 1; i <= local_N; i++) {
        for(int j = 0; j < N; j++) {
            A_local[i][j] = (double)(start_row + i -1) * N + j;
        }
    }

    
    if(rank == 0) {
        
        for(int j = 0; j < N; j++) {
            A_local[0][j] = 0.0; 
        }
    }
    else {
        
        for(int j = 0; j < N; j++) {
            A_local[0][j] = 0.0; 
        }
    }

    if(rank == P-1) {
        
        for(int j = 0; j < N; j++) {
            A_local[local_N+1][j] = 0.0; 
        }
    }
    else {
        
        for(int j = 0; j < N; j++) {
            A_local[local_N+1][j] = 0.0; 
        }
    }

    
    MPI_Request requests[4];
    int req_count = 0;

    
    if(rank > 0) {
        MPI_Isend(A_local[1], N, MPI_DOUBLE, rank-1, TAG_UP, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(A_local[0], N, MPI_DOUBLE, rank-1, TAG_DOWN, MPI_COMM_WORLD, &requests[req_count++]);
    }

    
    if(rank < P-1) {
        MPI_Isend(A_local[local_N], N, MPI_DOUBLE, rank+1, TAG_DOWN, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(A_local[local_N+1], N, MPI_DOUBLE, rank+1, TAG_UP, MPI_COMM_WORLD, &requests[req_count++]);
    }

    
    if(req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    
    *A_local_ptr = A_local;
    *B_local_ptr = B_local;
    *local_N_ptr = local_N;
    *start_row_ptr = start_row;
    return 0;
}

/**
 * 初始化按行块划分的 A_local
 */
void initialize_A_local_row(int local_N, int N, int start_row, double **A_local) {
    for(int i = 1; i <= local_N; i++) {
        for(int j = 0; j < N; j++) {
            A_local[i][j] = (double)(start_row + i -1) * N + j;
        }
    }
}

/**
 * 释放按行块划分的内存
 */
void free_memory_row(double **A_local, double **B_local, int local_N) {
    for(int i = 0; i < local_N + 2; i++) {
        free(A_local[i]);
        free(B_local[i]);
    }
    free(A_local);
    free(B_local);
}

/**
 * 收集并输出按行块连续划分的矩阵 A 和 B
 */
void gather_and_print_row(int N, int P, int rank, double **A_local, double **B_local, int local_N) {
    
    double *B_sendbuf = (double *)malloc(local_N * N * sizeof(double));
    for(int i = 0; i < local_N; i++) {
        for(int j = 0; j < N; j++) {
            B_sendbuf[i*N + j] = B_local[i+1][j];
        }
    }

    
    double *A_sendbuf = (double *)malloc(local_N * N * sizeof(double));
    for(int i = 0; i < local_N; i++) {
        for(int j = 0; j < N; j++) {
            A_sendbuf[i*N + j] = A_local[i+1][j];
        }
    }

    
    int *recvcounts_B = NULL;
    int *displs_B = NULL;
    double *B_global = NULL;
    if(rank == 0) {
        recvcounts_B = (int *)malloc(P * sizeof(int));
        displs_B = (int *)malloc(P * sizeof(int));
        int rows_per_proc = N / P;
        int remainder = N % P;
        for(int p = 0; p < P; p++) {
            recvcounts_B[p] = (rows_per_proc + (p < remainder ? 1 : 0)) * N;
        }
        displs_B[0] = 0;
        for(int p = 1; p < P; p++) {
            displs_B[p] = displs_B[p-1] + recvcounts_B[p-1];
        }
        B_global = (double *)malloc(N * N * sizeof(double));
    }

    MPI_Gatherv(B_sendbuf, local_N * N, MPI_DOUBLE,
                B_global, recvcounts_B, displs_B, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    
    int *recvcounts_A = NULL;
    int *displs_A = NULL;
    double *A_global = NULL;
    if(rank == 0) {
        recvcounts_A = (int *)malloc(P * sizeof(int));
        displs_A = (int *)malloc(P * sizeof(int));
        int rows_per_proc = N / P;
        int remainder = N % P;
        for(int p = 0; p < P; p++) {
            recvcounts_A[p] = (rows_per_proc + (p < remainder ? 1 : 0)) * N;
        }
        displs_A[0] = 0;
        for(int p = 1; p < P; p++) {
            displs_A[p] = displs_A[p-1] + recvcounts_A[p-1];
        }
        A_global = (double *)malloc(N * N * sizeof(double));
    }

    MPI_Gatherv(A_sendbuf, local_N * N, MPI_DOUBLE,
                A_global, recvcounts_A, displs_A, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    
    if(rank == 0) {
        printf("按行块连续划分结果示例:\n");
        printf("B[1][1] = %f\n", B_global[1*N +1]);
        printf("B[N-2][N-2] = %f\n", B_global[(N-2)*N + (N-2)]);
        
        printf("矩阵 A:\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                printf("%f ", A_global[i*N + j]);
            }
            printf("\n");
        }

        printf("矩阵 B:\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                printf("%f ", B_global[i*N + j]);
            }
            printf("\n");
        }

        
        free(recvcounts_B);
        free(displs_B);
        free(B_global);

        free(recvcounts_A);
        free(displs_A);
        free(A_global);
    }

    
    free(B_sendbuf);
    free(A_sendbuf);
}

/**
 * 棋盘式划分的实现
 * 返回 0 表示成功，-1 表示失败
 */
int checkerboard_partition(int N, int P, int rank, double ***A_local_ptr, double ***B_local_ptr, int *local_rows_ptr, int *local_cols_ptr, int *start_row_ptr, int *start_col_ptr, MPI_Comm *grid_comm_ptr) {
    
    int q = (int)sqrt(P);
    if(q * q != P) {
        if(rank == 0) {
            printf("Number of processes must be a perfect square for checkerboard partitioning.\n");
        }
        return -1;
    }

    
    int dims[2] = {q, q};
    int periods[2] = {0, 0}; 
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    if(grid_comm == MPI_COMM_NULL) {
        printf("Process %d: Failed to create Cartesian grid.\n", rank);
        return -1;
    }

    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    
    int up, down, left, right;
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down); 
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right); 

    
    int rows_per_proc = N / q;
    int row_remainder = N % q;
    int cols_per_proc = N / q;
    int col_remainder = N % q;

    int local_rows = rows_per_proc + (my_row < row_remainder ? 1 : 0);
    int local_cols = cols_per_proc + (my_col < col_remainder ? 1 : 0);
    int start_row = my_row * rows_per_proc + (my_row < row_remainder ? my_row : row_remainder);
    int start_col = my_col * cols_per_proc + (my_col < col_remainder ? my_col : col_remainder);

    
    double **A_local = (double **)malloc((local_rows + 2) * sizeof(double *));
    double **B_local = (double **)malloc((local_rows + 2) * sizeof(double *));
    if(A_local == NULL || B_local == NULL) {
        printf("Process %d: Memory allocation failed.\n", rank);
        return -1;
    }
    for(int i = 0; i < local_rows + 2; i++) {
        A_local[i] = (double *)malloc((local_cols + 2) * sizeof(double));
        B_local[i] = (double *)malloc((local_cols + 2) * sizeof(double));
        if(A_local[i] == NULL || B_local[i] == NULL) {
            printf("Process %d: Memory allocation failed.\n", rank);
            return -1;
        }
    }

    
    
    for(int i = 1; i <= local_rows; i++) {
        for(int j = 1; j <= local_cols; j++) {
            A_local[i][j] = (double)(start_row + i -1) * N + (start_col + j -1);
        }
    }

    
    for(int i = 0; i < local_rows + 2; i++) {
        A_local[i][0] = 0.0; 
        A_local[i][local_cols+1] = 0.0; 
    }
    for(int j = 0; j < local_cols + 2; j++) {
        A_local[0][j] = 0.0; 
        A_local[local_rows+1][j] = 0.0; 
    }

    
    MPI_Request requests[8];
    int req_count = 0;
    
    double *send_left = NULL, *recv_right = NULL;
    double *send_right = NULL, *recv_left = NULL;

    
    if(up != MPI_PROC_NULL) {
        MPI_Isend(&A_local[1][1], local_cols, MPI_DOUBLE, up, TAG_UP, grid_comm, &requests[req_count++]);
        MPI_Irecv(&A_local[0][1], local_cols, MPI_DOUBLE, up, TAG_DOWN, grid_comm, &requests[req_count++]);
    }

    
    if(down != MPI_PROC_NULL) {
        MPI_Isend(&A_local[local_rows][1], local_cols, MPI_DOUBLE, down, TAG_DOWN, grid_comm, &requests[req_count++]);
        MPI_Irecv(&A_local[local_rows+1][1], local_cols, MPI_DOUBLE, down, TAG_UP, grid_comm, &requests[req_count++]);
    }

    
    if(left != MPI_PROC_NULL) {
        
        send_left = (double *)malloc(local_rows * sizeof(double));
        for(int i = 0; i < local_rows; i++) {
            send_left[i] = A_local[i+1][1];
        }
        MPI_Isend(send_left, local_rows, MPI_DOUBLE, left, TAG_LEFT, grid_comm, &requests[req_count++]);

        
        recv_right = (double *)malloc(local_rows * sizeof(double));
        MPI_Irecv(recv_right, local_rows, MPI_DOUBLE, left, TAG_RIGHT, grid_comm, &requests[req_count++]);
    }

    
    if(right != MPI_PROC_NULL) {
        
        send_right = (double *)malloc(local_rows * sizeof(double));
        for(int i = 0; i < local_rows; i++) {
            send_right[i] = A_local[i+1][local_cols];
        }
        MPI_Isend(send_right, local_rows, MPI_DOUBLE, right, TAG_RIGHT, grid_comm, &requests[req_count++]);

        
        recv_left = (double *)malloc(local_rows * sizeof(double));
        MPI_Irecv(recv_left, local_rows, MPI_DOUBLE, right, TAG_LEFT, grid_comm, &requests[req_count++]);
    }

    
    if(req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    
    if(left != MPI_PROC_NULL && recv_right != NULL) {
        for(int i = 0; i < local_rows; i++) {
            A_local[i+1][local_cols+1] = recv_right[i];
        }
        free(send_left);
        free(recv_right);
    }

    if(right != MPI_PROC_NULL && recv_left != NULL) {
        for(int i = 0; i < local_rows; i++) {
            A_local[i+1][0] = recv_left[i];
        }
        free(send_right);
        free(recv_left);
    }

    
    *A_local_ptr = A_local;
    *B_local_ptr = B_local;
    *local_rows_ptr = local_rows;
    *local_cols_ptr = local_cols;
    *start_row_ptr = start_row;
    *start_col_ptr = start_col;
    *grid_comm_ptr = grid_comm;
    return 0;
}

/**
 * 初始化棋盘式划分的 A_local
 */
void initialize_A_local_checkerboard(int local_rows, int local_cols, int start_row, int start_col, double **A_local, int N) {
    for(int i = 1; i <= local_rows; i++) {
        for(int j = 1; j <= local_cols; j++) {
            A_local[i][j] = (double)(start_row + i -1) * N + (start_col + j -1);
        }
    }
}

/**
 * 释放棋盘式划分的内存
 */
void free_memory_checkerboard(double **A_local, double **B_local, int local_rows) {
    for(int i = 0; i < local_rows + 2; i++) {
        free(A_local[i]);
        free(B_local[i]);
    }
    free(A_local);
    free(B_local);
}

/**
 * 收集并输出棋盘式划分的矩阵 A 和 B
 */
void gather_and_print_checkerboard(int N, int P, int rank, double **A_local, double **B_local, int local_rows, int local_cols, MPI_Comm grid_comm) {
    
    double *B_sendbuf = (double *)malloc(local_rows * local_cols * sizeof(double));
    for(int i = 0; i < local_rows; i++) {
        for(int j = 0; j < local_cols; j++) {
            B_sendbuf[i*local_cols + j] = B_local[i+1][j+1];
        }
    }

    
    double *A_sendbuf = (double *)malloc(local_rows * local_cols * sizeof(double));
    for(int i = 0; i < local_rows; i++) {
        for(int j = 0; j < local_cols; j++) {
            A_sendbuf[i*local_cols + j] = A_local[i+1][j+1];
        }
    }

    
    int *recvcounts_B = NULL;
    int *displs_B = NULL;
    double *B_global = NULL;
    if(rank == 0) {
        recvcounts_B = (int *)malloc(P * sizeof(int));
        displs_B = (int *)malloc(P * sizeof(int));
        
        for(int p = 0; p < P; p++) {
            int coords_p[2];
            MPI_Cart_coords(grid_comm, p, 2, coords_p);
            int pr = coords_p[0];
            int pc = coords_p[1];
            int q = (int)sqrt(P);
            int rows_p = N / q + (pr < (N % q) ? 1 : 0);
            int cols_p = N / q + (pc < (N % q) ? 1 : 0);
            recvcounts_B[p] = rows_p * cols_p;
        }

        
        displs_B[0] = 0;
        for(int p = 1; p < P; p++) {
            displs_B[p] = displs_B[p-1] + recvcounts_B[p-1];
        }

        B_global = (double *)malloc(N * N * sizeof(double));
    }

    MPI_Gatherv(B_sendbuf, local_rows * local_cols, MPI_DOUBLE,
                B_global, recvcounts_B, displs_B, MPI_DOUBLE,
                0, grid_comm);

    
    int *recvcounts_A = NULL;
    int *displs_A = NULL;
    double *A_global = NULL;
    if(rank == 0) {
        recvcounts_A = (int *)malloc(P * sizeof(int));
        displs_A = (int *)malloc(P * sizeof(int));
        
        for(int p = 0; p < P; p++) {
            int coords_p[2];
            MPI_Cart_coords(grid_comm, p, 2, coords_p);
            int pr = coords_p[0];
            int pc = coords_p[1];
            int q = (int)sqrt(P);
            int rows_p = N / q + (pr < (N % q) ? 1 : 0);
            int cols_p = N / q + (pc < (N % q) ? 1 : 0);
            recvcounts_A[p] = rows_p * cols_p;
        }

        
        displs_A[0] = 0;
        for(int p = 1; p < P; p++) {
            displs_A[p] = displs_A[p-1] + recvcounts_A[p-1];
        }

        A_global = (double *)malloc(N * N * sizeof(double));
    }

    MPI_Gatherv(A_sendbuf, local_rows * local_cols, MPI_DOUBLE,
                A_global, recvcounts_A, displs_A, MPI_DOUBLE,
                0, grid_comm);

    
    if(rank == 0) {
        printf("棋盘式划分结果示例:\n");
        printf("B[1][1] = %f\n", B_global[1*N +1]);
        printf("B[N-2][N-2] = %f\n", B_global[(N-2)*N + (N-2)]);
        
        printf("矩阵 A:\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                printf("%f ", A_global[i*N + j]);
            }
            printf("\n");
        }

        printf("矩阵 B:\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                printf("%f ", B_global[i*N + j]);
            }
            printf("\n");
        }

        
        free(recvcounts_B);
        free(displs_B);
        free(B_global);

        free(recvcounts_A);
        free(displs_A);
        free(A_global);
    }

    
    free(B_sendbuf);
    free(A_sendbuf);
}
