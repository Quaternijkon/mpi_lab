// #include <mpi.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

// #define N 4 
// #define ROOT 0

// void initialize_matrix(double* matrix, int n) {
//     for (int i = 0; i < n * n; i++) {
//         matrix[i] = i + 1; 
//     }
// }

// void print_matrix(double* matrix, int n) {
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             printf("%5.1f ", matrix[i * n + j]);
//         }
//         printf("\n");
//     }
// }

// int main(int argc, char** argv) {
//     int rank, size;
//     int grid_size, block_size;
//     int coords[2], periods[2] = {1, 1}; 
//     MPI_Comm grid_comm, row_comm, col_comm;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     grid_size = (int)sqrt(size);
//     if (grid_size * grid_size != size) {
//         if (rank == ROOT) {
//             printf("进程数必须是完全平方数！\n");
//         }
//         MPI_Finalize();
//         return -1;
//     }

//     block_size = N / grid_size;
//     if (N % grid_size != 0) {
//         if (rank == ROOT) {
//             printf("矩阵维度必须能被网格大小整除！\n");
//         }
//         MPI_Finalize();
//         return -1;
//     }

//     // 修正：使用可修改的数组 dims
//     int dims[2] = {grid_size, grid_size};
//     MPI_Dims_create(size, 2, dims);
//     MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
//     MPI_Cart_coords(grid_comm, rank, 2, coords);

//     MPI_Comm_split(grid_comm, coords[0], coords[1], &row_comm);
//     MPI_Comm_split(grid_comm, coords[1], coords[0], &col_comm);

//     double* A = NULL;
//     double* B = NULL;
//     double* C = NULL;
//     if (rank == ROOT) {
//         A = malloc(N * N * sizeof(double));
//         B = malloc(N * N * sizeof(double));
//         C = malloc(N * N * sizeof(double));
//         initialize_matrix(A, N);
//         initialize_matrix(B, N);
//         for (int i = 0; i < N * N; i++) {
//             C[i] = 0.0;
//         }
//         printf("矩阵 A:\n");
//         print_matrix(A, N);
//         printf("矩阵 B:\n");
//         print_matrix(B, N);
//     }

//     double* local_A = malloc(block_size * block_size * sizeof(double));
//     double* local_B = malloc(block_size * block_size * sizeof(double));
//     double* local_C = malloc(block_size * block_size * sizeof(double));
//     for (int i = 0; i < block_size * block_size; i++) {
//         local_C[i] = 0.0;
//     }

//     double* temp_A = malloc(block_size * block_size * sizeof(double));
//     MPI_Scatter(A, block_size * block_size, MPI_DOUBLE, local_A, block_size * block_size, MPI_DOUBLE, ROOT, grid_comm);
//     MPI_Scatter(B, block_size * block_size, MPI_DOUBLE, local_B, block_size * block_size, MPI_DOUBLE, ROOT, grid_comm);

//     for (int step = 0; step < grid_size; step++) {
//         int pivot = (coords[0] + step) % grid_size;
//         if (coords[1] == pivot) {
//             for (int i = 0; i < block_size * block_size; i++) {
//                 temp_A[i] = local_A[i];
//             }
//         }

//         MPI_Bcast(temp_A, block_size * block_size, MPI_DOUBLE, pivot, row_comm);

//         for (int i = 0; i < block_size; i++) {
//             for (int j = 0; j < block_size; j++) {
//                 for (int k = 0; k < block_size; k++) {
//                     local_C[i * block_size + j] += temp_A[i * block_size + k] * local_B[k * block_size + j];
//                 }
//             }
//         }

//         MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE,
//                              (coords[0] + 1) % grid_size, 0,
//                              (coords[0] - 1 + grid_size) % grid_size, 0,
//                              col_comm, MPI_STATUS_IGNORE);
//     }

//     MPI_Gather(local_C, block_size * block_size, MPI_DOUBLE, C, block_size * block_size, MPI_DOUBLE, ROOT, grid_comm);

//     if (rank == ROOT) {
//         printf("结果矩阵 C:\n");
//         print_matrix(C, N);
//     }

//     if (rank == ROOT) {
//         free(A);
//         free(B);
//         free(C);
//     }
//     free(local_A);
//     free(local_B);
//     free(local_C);
//     free(temp_A);

//     MPI_Finalize();
//     return 0;
// }
