#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef N
#define N 7
#endif

#define INDEX(i, j) (((i)*N)+(j))


void fill_matrix(double *a, int num) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            
            a[INDEX(i, j)] = i + j;  
        }
    }
}

void comp(double *A, double *B, int a, int b) {
    for (int i = 0; i <= a; i++) {
        for (int j = 0; j <= b; j++) {
            B[INDEX(i, j)] = (A[INDEX(i-1, j)] + A[INDEX(i, j+1)] + A[INDEX(i+1, j)] + A[INDEX(i, j-1)]) / 4.0;
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

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%lf ", matrix[INDEX(i, j)]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
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
    int alloc_num = (a+1)*(b+1)*num_procs;
    A = (double*)malloc(alloc_num * sizeof(double));
    B = (double*)malloc(alloc_num * sizeof(double));
    B2 = (double*)malloc(alloc_num * sizeof(double));

    
    if (id_procs == 0) {
        fill_matrix(A, N * N);
        comp(A, B2, N-1, N-1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Type_vector(a+2, b+2, N, MPI_DOUBLE, &SubMat);
    MPI_Type_commit(&SubMat);

    if (id_procs == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == 0 && j == 0)
                    continue;
                MPI_Send(A + i * a * N + b * j, 1, SubMat, j + cols * i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(A, 1, SubMat, 0, 0, MPI_COMM_WORLD, &status);
    }

    
    comp(A, B, a, b);

    
    MPI_Datatype SubMat_B;
    MPI_Type_vector(a, b, N, MPI_DOUBLE, &SubMat_B);
    MPI_Type_commit(&SubMat_B);
    if (id_procs == 0) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == 0 && j == 0)
                    continue;
                MPI_Recv(&B[INDEX(a * i + 1, b * j + 1)], 1, SubMat_B, i * cols + j, 1, MPI_COMM_WORLD, &status);
            }
        }
    } else {
        int x = id_procs / cols;
        int y = id_procs % cols;
        MPI_Send(&B[INDEX(1, 1)], 1, SubMat_B, 0, 1, MPI_COMM_WORLD);
    }

    if (id_procs == 0) {
        if (check(B, B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error!\n");
        }

        
        printf("\nMatrix A:\n");
        print_matrix(A, N, N);

        printf("\nMatrix B (after computation):\n");
        print_matrix(B, N, N);
    }

    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
