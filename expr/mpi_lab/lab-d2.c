#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 4  
#define ROOT 0

void matmul(float *A, float *B, float *C, int size, int rank, int p) {
    
    int sqrt_p = (int) sqrt(p);
    int block_size = size / sqrt_p;
    float *local_A = (float *)malloc(block_size * size * sizeof(float));
    float *local_B = (float *)malloc(block_size * size * sizeof(float));
    float *local_C = (float *)malloc(block_size * size * sizeof(float));

    
    MPI_Scatter(A, block_size * size, MPI_FLOAT, local_A, block_size * size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(B, block_size * size, MPI_FLOAT, local_B, block_size * size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    
    for (int step = 0; step < sqrt_p; step++) {
        
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_C[i * block_size + j] = 0;
                for (int k = 0; k < size; k++) {
                    local_C[i * block_size + j] += local_A[i * size + k] * local_B[k * size + j];
                }
            }
        }

        
        int row_shift = (rank / sqrt_p + step) % sqrt_p;
        int col_shift = (rank % sqrt_p + step) % sqrt_p;
        
        
        MPI_Sendrecv(local_A, block_size * size, MPI_FLOAT, 
                     (rank + row_shift) % sqrt_p, 0, 
                     local_A, block_size * size, MPI_FLOAT, 
                     (rank + row_shift) % sqrt_p, 0, MPI_COMM_WORLD);
        
        MPI_Sendrecv(local_B, block_size * size, MPI_FLOAT,
                     (rank + col_shift) % sqrt_p, 0,
                     local_B, block_size * size, MPI_FLOAT,
                     (rank + col_shift) % sqrt_p, 0, MPI_COMM_WORLD);
    }
    
    
    MPI_Gather(local_C, block_size * size, MPI_FLOAT, C, block_size * size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    free(local_A);
    free(local_B);
    free(local_C);
}

int main(int argc, char *argv[]) {
    int rank, size;
    float *A, *B, *C;
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == ROOT) {
            printf("This program requires 4 MPI processes.\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    
    if (rank == ROOT) {
        A = (float *)malloc(N * N * sizeof(float));
        B = (float *)malloc(N * N * sizeof(float));
        C = (float *)malloc(N * N * sizeof(float));
        
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
            }
        }
    }

    
    matmul(A, B, C, N, rank, size);

    
    if (rank == ROOT) {
        printf("Matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", C[i * N + j]);
            }
            printf("\n");
        }

        
        free(A);
        free(B);
        free(C);
    }

    
    MPI_Finalize();
    return 0;
}
