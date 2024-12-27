#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void initialize_matrices(double *A, double *B, int n, int block_size, int rank, int size) {
    for(int i = 0; i < block_size * block_size; i++) {
        A[i] = 1.0; 
        B[i] = 1.0;
    }
}


void print_matrix(double *matrix, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%lf ", matrix[i * n + j]);
        }
        printf("\n");
    }
}


void multiply_add(double *A, double *B, double *C, int block_size) {
    for(int i = 0; i < block_size; i++) {
        for(int j = 0; j < block_size; j++) {
            for(int k = 0; k < block_size; k++) {
                C[i * block_size + j] += A[i * block_size + k] * B[k * block_size + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 8; 
    int sqrt_p;
    int block_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
    sqrt_p = (int)sqrt((double)size);
    if(sqrt_p * sqrt_p != size) {
        if(rank == 0) {
            printf("进程数必须为完全平方数。\n");
        }
        MPI_Finalize();
        return -1;
    }
    
    
    if(n % sqrt_p != 0) {
        if(rank == 0) {
            printf("矩阵大小必须能被 sqrt(p) 整除。\n");
        }
        MPI_Finalize();
        return -1;
    }
    
    block_size = n / sqrt_p;
    
    
    MPI_Comm grid_comm;
    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {1, 1}; 
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    
    
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);
    
    
    double *A_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *B_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *C_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *A_temp = (double *)malloc(block_size * block_size * sizeof(double));
    
    
    initialize_matrices(A_block, B_block, n, block_size, rank, size);
    
    
    for(int i = 0; i < block_size * block_size; i++) {
        C_block[i] = 0.0;
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int step = 0; step < sqrt_p; step++) {
        int root = (my_row + step) % sqrt_p;
        
        if(root == my_col) {
            
            MPI_Bcast(A_block, block_size * block_size, MPI_DOUBLE, root, row_comm);
            
            for(int i = 0; i < block_size * block_size; i++) {
                A_temp[i] = A_block[i];
            }
        } else {
            
            MPI_Bcast(A_temp, block_size * block_size, MPI_DOUBLE, root, row_comm);
        }
        
        
        multiply_add(A_temp, B_block, C_block, block_size);
        
        
        int source, dest;
        MPI_Cart_shift(grid_comm, 1, -1, &source, &dest);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE, dest, 0, source, 0, grid_comm, MPI_STATUS_IGNORE);
    }
    
    
    double *C = NULL;
    if(rank == 0) {
        C = (double *)malloc(n * n * sizeof(double));
    }
    
    
    MPI_Datatype block_type, block_type_resized;
    MPI_Type_vector(block_size, block_size, sqrt_p * block_size, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &block_type_resized);
    MPI_Type_commit(&block_type_resized);
    
    int *displs = NULL;
    int *recvcounts = NULL;
    if(rank == 0) {
        displs = (int *)malloc(size * sizeof(int));
        recvcounts = (int *)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            displs[i] = i;
            recvcounts[i] = 1;
        }
    }
    
    MPI_Gather(C_block, block_size * block_size, MPI_DOUBLE, C, block_size * block_size, MPI_DOUBLE, 0, grid_comm);
    
    
    if(rank == 0) {
        printf("结果矩阵 C:\n");
        print_matrix(C, n);
        free(C);
        free(displs);
        free(recvcounts);
    }
    
    
    free(A_block);
    free(B_block);
    free(C_block);
    free(A_temp);
    MPI_Type_free(&block_type_resized);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}
