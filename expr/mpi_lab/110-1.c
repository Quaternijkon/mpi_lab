#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef N
#define N 7
#endif

#define INDEX(i, j) (((i)*N)+(j))


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


void comp(double *A, double *B, int num) {
    
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
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    B2 = (double*)malloc(N * N * sizeof(double));

    int id_procs, num_procs, num_1;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);

    num_1 = num_procs - 1;

    
    if (id_procs == num_1) {
        random_array(A, N * N);
        comp(A, B2, N * N);
    }

    MPI_Barrier(MPI_COMM_WORLD);

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
        MPI_Recv(&A[INDEX(ctn * 3, 0)], 3 * N, MPI_DOUBLE, num_1, ctn, MPI_COMM_WORLD, &status);
        ctn++;
    }

    
    if (id_procs != num_1) {
        for (int i = 0; i <= ctn; i++) {
            for (int j = 0; j < N; j++) {
                
                B[INDEX(i, j)] = (A[INDEX(i - 1 + N, j)] + A[INDEX(i, j + 1)] + A[INDEX(i + 1, j)] + A[INDEX(i, j - 1 + N)]) / 4.0;
            }
        }
    }

    
    for (int i = 0; i < N - 2; i++) {
        if (id_procs == num_1) {
            int src = i % num_1;
            MPI_Recv(&B[INDEX(i + 1, 1)], N - 2, MPI_DOUBLE, src, i / num_1 + N, MPI_COMM_WORLD, &status);
        } else {
            for (int j = 0; j < ctn; j++)
                MPI_Send(&B[INDEX(j + 1, 1)], N - 2, MPI_DOUBLE, num_1, j + N, MPI_COMM_WORLD);
        }
    }

    
    if (id_procs == num_1) {
        if (check(B, B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error Occurred!\n");
        }

        
        printf("\nMatrix A (initialized with indices):\n");
        print_matrix(A);

        printf("\nMatrix B (calculated result):\n");
        print_matrix(B);
    }

    
    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
