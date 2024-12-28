// #include <stdlib.h>
// #include <stdio.h>
// #include <math.h>
// #include <mpi.h>
// #include <time.h> // 需要包含以使用 time()

// typedef struct {
//     MPI_Comm grid_comm; /* 全局网格通信器 */
//     MPI_Comm row_comm;  /* 行通信器 */
//     MPI_Comm col_comm;  /* 列通信器 */
//     int n_proc;         /* 进程数 */
//     int grid_dim;       /* 网格的维度, = sqrt(n_proc) */
//     int my_row;         /* 当前进程所在的行 */
//     int my_col;         /* 当前进程所在的列 */
//     int my_rank;        /* 当前进程的rank */
// } GridInfo;

// void grid_init(GridInfo *grid);
// void FoxAlgorithm(double *A, double *B, double *C, int size, GridInfo *grid);
// void matrix_creation(double **pA, double **pB, double **pC, int size);
// void matrix_init(double *A, double *B, int size, int sup);
// void matrix_print(double *A, int size);

// void grid_init(GridInfo *grid)
// {
//     int old_rank;
//     int dimensions[2];
//     int wrap_around[2];
//     int coordinates[2];
//     int free_coords[2];

//     /* 获取全局的进程信息 */
//     MPI_Comm_size(MPI_COMM_WORLD, &(grid->n_proc));
//     MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

//     grid->grid_dim = (int)sqrt(grid->n_proc);
//     /* 错误检查：进程数应该是完全平方数 */
//     if (grid->grid_dim * grid->grid_dim != grid->n_proc) {
//         if (old_rank == 0) {
//             printf("[!] 进程数不是完全平方数!\n");
//         }
//         MPI_Finalize();
//         exit(-1);
//     }

//     /* 设置网格维度 */
//     dimensions[0] = dimensions[1] = grid->grid_dim;
//     wrap_around[0] = wrap_around[1] = 1; // 是否环绕

//     MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, wrap_around, 1, &(grid->grid_comm));
//     MPI_Comm_rank(grid->grid_comm, &(grid->my_rank));
//     MPI_Cart_coords(grid->grid_comm, grid->my_rank, 2, coordinates);
//     grid->my_row = coordinates[0];
//     grid->my_col = coordinates[1];

//     /* 创建行通信器 */
//     free_coords[0] = 0; // 保留第0维度（行）
//     free_coords[1] = 1; // 禁用第1维度（列）
//     MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->row_comm));

//     /* 创建列通信器 */
//     free_coords[0] = 1; // 禁用第0维度（行）
//     free_coords[1] = 0; // 保留第1维度（列）
//     MPI_Cart_sub(grid->grid_comm, free_coords, &(grid->col_comm));
// }

// void matrix_creation(double **pA, double **pB, double **pC, int size)
// {
//     *pA = (double *)malloc(size * size * sizeof(double));
//     *pB = (double *)malloc(size * size * sizeof(double));
//     *pC = (double *)calloc(size * size, sizeof(double));
// }

// void matrix_init(double *A, double *B, int size, int sup)
// {
//     srand(time(NULL)); // 初始化随机数种子
//     for (int i = 0; i < size * size; ++i) {
//         A[i] = rand() % sup + 1;
//         B[i] = rand() % sup + 1;
//     }
// }

// void matrix_print(double *A, int size)
// {
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             printf("%6.2f ", A[i * size + j]);
//         }
//         printf("\n");
//     }
// }

// void matrix_dot(double *A, double *B, double *C, int size)
// {
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             for (int k = 0; k < size; ++k) {
//                 C[i * size + j] += A[i * size + k] * B[k * size + j];
//             }
//         }
//     }
// }

// void FoxAlgorithm(double *A, double *B, double *C, int size, GridInfo *grid)
// {
//     double *buff_A = (double*)calloc(size * size, sizeof(double));
//     MPI_Status status;
//     int root;
//     int src = (grid->my_row + 1) % grid->grid_dim;
//     int dst = (grid->my_row - 1 + grid->grid_dim) % grid->grid_dim;

//     /* Fox算法的核心循环 */
//     for (int stage = 0; stage < grid->grid_dim; ++stage) {
//         root = (grid->my_row + stage) % grid->grid_dim;
//         if (root == grid->my_col) {
//             MPI_Bcast(A, size * size, MPI_DOUBLE, root, grid->row_comm);
//             matrix_dot(A, B, C, size);
//         } else {
//             MPI_Bcast(buff_A, size * size, MPI_DOUBLE, root, grid->row_comm);
//             matrix_dot(buff_A, B, C, size);
//         }
//         MPI_Sendrecv_replace(B, size * size, MPI_DOUBLE, dst, 0, src, 0, grid->col_comm, &status);
//     }

//     free(buff_A);
// }

// int main(int argc, char **argv)
// {
//     double *pA, *pB, *pC;
//     double *local_pA, *local_pB, *local_pC;
//     int matrix_size = 100;

//     if (argc == 2) {
//         sscanf(argv[1], "%d", &matrix_size);
//     }

//     MPI_Init(&argc, &argv);

//     GridInfo grid;
//     grid_init(&grid);

//     /* 错误检查：确保矩阵大小是网格维度的整数倍 */
//     if (matrix_size % grid.grid_dim != 0) {
//         if (grid.my_rank == 0) {
//             printf("[!] matrix_size mod sqrt(n_processes) != 0 !\n");
//         }
//         MPI_Finalize();
//         exit(-1);
//     }

//     if (grid.my_rank == 0) {
//         matrix_creation(&pA, &pB, &pC, matrix_size);
//         matrix_init(pA, pB, matrix_size, 10);

//         printf("Matrix A (size=%d):\n", matrix_size);
//         matrix_print(pA, matrix_size);
//         printf("\nMatrix B (size=%d):\n", matrix_size);
//         matrix_print(pB, matrix_size);
//         printf("\n");
//     }

//     int local_matrix_size = matrix_size / grid.grid_dim;
//     matrix_creation(&local_pA, &local_pB, &local_pC, local_matrix_size);

//     /* MPI 数据类型设置 */
//     MPI_Datatype blocktype, type;
//     int array_size[2] = {matrix_size, matrix_size};
//     int subarray_sizes[2] = {local_matrix_size, local_matrix_size};
//     int array_start[2] = {0, 0};
//     MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start,
//                              MPI_ORDER_C, MPI_DOUBLE, &blocktype);
//     MPI_Type_create_resized(blocktype, 0, local_matrix_size * sizeof(double), &type);
//     MPI_Type_commit(&type);

//     int displs[grid.n_proc];
//     int sendcounts[grid.n_proc];
//     if (grid.my_rank == 0) {
//         for (int i = 0; i < grid.n_proc; ++i) {
//             sendcounts[i] = 1;
//         }
//         int disp = 0;
//         for (int i = 0; i < grid.grid_dim; ++i) {
//             for (int j = 0; j < grid.grid_dim; ++j) {
//                 displs[i * grid.grid_dim + j] = disp;
//                 disp += 1;
//             }
//             disp += (local_matrix_size - 1) * grid.grid_dim;
//         }
//     }

//     /* 分发矩阵A和B */
//     MPI_Scatterv(pA, sendcounts, displs, type, local_pA,
//                  local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     MPI_Scatterv(pB, sendcounts, displs, type, local_pB,
//                  local_matrix_size * local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     /* 调用Fox算法进行矩阵乘法 */
//     FoxAlgorithm(local_pA, local_pB, local_pC, local_matrix_size, &grid);

//     /* 收集矩阵C */
//     MPI_Gatherv(local_pC, local_matrix_size * local_matrix_size, MPI_DOUBLE, pC, sendcounts, displs, type, 0, MPI_COMM_WORLD);

//     /* 输出结果 */
//     if (grid.my_rank == 0) {
//         printf("Matrix multiplication completed\n\n");
//         printf("Matrix C (result of A * B):\n");
//         matrix_print(pC, matrix_size);
//     }

//     /* 清理内存 */
//     if (grid.my_rank == 0) {
//         free(pA);
//         free(pB);
//         free(pC);
//     }
//     free(local_pA);
//     free(local_pB);
//     free(local_pC);

//     MPI_Finalize();
//     return 0;
// }
