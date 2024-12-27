#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024 

int main(int argc, char *argv[]) {
    int rank, size;
    int dims[2], periods[2] = {1, 1}; 
    int coords[2];
    MPI_Comm grid_comm, row_comm, col_comm;
    int sqrt_p;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    sqrt_p = (int)sqrt((double)size);
    if (sqrt_p * sqrt_p != size) {
        if (rank == 0) {
            printf("Number of processes must be a perfect square.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    
    dims[0] = dims[1] = sqrt_p;

    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);

    
    int block_size = N / sqrt_p;

    
    double *A_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *B_block = (double *)malloc(block_size * block_size * sizeof(double));
    double *C_block = (double *)calloc(block_size * block_size, sizeof(double)); 
    double *A_temp = (double *)malloc(block_size * block_size * sizeof(double));

    
    for (i = 0; i < block_size * block_size; i++) {
        A_block[i] = 1.0; 
        B_block[i] = 1.0; 
    }

    
    for (k = 0; k < sqrt_p; k++) {
        int root = (my_row + k) % sqrt_p; 

        
        if (root == my_col) {
            for (i = 0; i < block_size * block_size; i++) {
                A_temp[i] = A_block[i]; 
            }
        }
        MPI_Bcast(A_temp, block_size * block_size, MPI_DOUBLE, root, row_comm);

        
        for (i = 0; i < block_size; i++) {
            for (j = 0; j < block_size; j++) {
                double sum = 0.0;
                for (int l = 0; l < block_size; l++) {
                    sum += A_temp[i * block_size + l] * B_block[l * block_size + j];
                }
                C_block[i * block_size + j] += sum;
            }
        }

        
        int src, dest;
        MPI_Cart_shift(grid_comm, 0, -1, &src, &dest);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE,
                             dest, 0, src, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    
    double *C_result = NULL;
    if (rank == 0) {
        C_result = (double *)malloc(N * N * sizeof(double));
    }
    MPI_Gather(C_block, block_size * block_size, MPI_DOUBLE,
               C_result, block_size * block_size, MPI_DOUBLE, 0, grid_comm);

    
    if (rank == 0) {
        printf("C_result[0][0] = %f\n", C_result[0]);
        free(C_result);
    }

    
    free(A_block);
    free(B_block);
    free(C_block);
    free(A_temp);

    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}
