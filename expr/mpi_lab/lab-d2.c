#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ROOT 0


void initialize_matrices(double *A, double *B, int n, int proc_row, int proc_col, int grid_size) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            
            
            
            int global_i = proc_row * n + i;
            int global_j = proc_col * n + j;
            A[i * n + j] = (double)(global_i + global_j);
            B[i * n + j] = (double)(global_i - global_j);
        }
    }
}


void print_matrix(double *matrix, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%8.2lf ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 8; 
    double start_time, end_time;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    for(int i = 1; i < argc; i++) {
        if(strncmp(argv[i], "-N=", 3) == 0) {
            N = atoi(argv[i] + 3);
        }
    }

    
    int grid_dim = (int)sqrt((double)size);
    if(grid_dim * grid_dim != size) {
        if(rank == ROOT) {
            printf("Error: Number of processes must be a perfect square.\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    if(N % grid_dim != 0) {
        if(rank == ROOT) {
            printf("Error: Matrix size N must be divisible by sqrt(p).\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int n = N / grid_dim; 

    
    int dims[2] = {grid_dim, grid_dim};
    int periods[2] = {1, 1}; 
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int proc_row = coords[0];
    int proc_col = coords[1];

    
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, proc_row, proc_col, &row_comm);
    MPI_Comm_split(grid_comm, proc_col, proc_row, &col_comm);

    
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)calloc(n * n, sizeof(double)); 
    double *A_temp = (double*)malloc(n * n * sizeof(double));

    if(A == NULL || B == NULL || C == NULL || A_temp == NULL) {
        printf("Error: Process %d cannot allocate memory.\n", rank);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    initialize_matrices(A, B, n, proc_row, proc_col, grid_dim);

    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    
    for(int step = 0; step < grid_dim; step++) {
        int root = (proc_row + step) % grid_dim;

        
        int broadcast_rank;
        if(root == proc_col) {
            
            memcpy(A_temp, A, n * n * sizeof(double));
            broadcast_rank = proc_col;
        }

        
        MPI_Bcast(A_temp, n * n, MPI_DOUBLE, root, row_comm);

        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int k = 0; k < n; k++) {
                    C[i * n + j] += A_temp[i * n + k] * B[k * n + j];
                }
            }
        }

        
        int src, dest;
        MPI_Cart_shift(grid_comm, 1, -1, &src, &dest);
        MPI_Sendrecv_replace(B, n * n, MPI_DOUBLE, dest, 0, src, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    
    double *C_final = NULL;
    if(rank == ROOT) {
        C_final = (double*)malloc(N * N * sizeof(double));
        if(C_final == NULL) {
            printf("Error: Root process cannot allocate memory for C_final.\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    
    MPI_Datatype block_type, block_type_resized;
    MPI_Type_vector(n, n, grid_dim * n, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &block_type_resized);
    MPI_Type_commit(&block_type_resized);

    
    MPI_Gather(C, n * n, MPI_DOUBLE, C_final, 1, block_type_resized, ROOT, grid_comm);

    
    if(rank == ROOT) {
        printf("最终结果矩阵 C:\n");
        print_matrix(C_final, N);
        printf("\n");
        printf("耗时: %lf 秒\n", end_time - start_time);
        free(C_final);
    }

    
    MPI_Type_free(&block_type_resized);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    free(A);
    free(B);
    free(C);
    free(A_temp);

    MPI_Finalize();
    return 0;
}
