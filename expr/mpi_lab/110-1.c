#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define INDEX(i, j, N) (((i)*N)+(j)) 


void print_matrix(const char *name, double *matrix, int n) {
    printf("Matrix %s:\n", name);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%6.2lf ", matrix[INDEX(i, j, n)]);
        }
        printf("\n");
    }
    printf("\n");
}


void random_array(double *a, int num) {
    srand(time(NULL)); 
    for(int i = 0; i < num; i++) {
        a[i] = rand() % 100;
    }
}


void comp(double *A, double *B, int N) {
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {
            B[INDEX(i, j, N)] = (A[INDEX(i-1, j, N)] + A[INDEX(i, j+1, N)] + A[INDEX(i+1, j, N)] + A[INDEX(i, j-1, N)]) / 4.0;
        }
    }
}


int check(double *B, double *C, int N) {
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {
            if (fabs(B[INDEX(i, j, N)] - C[INDEX(i, j, N)]) >= 1e-2) {
                printf("B[%d,%d] = %lf not %lf!\n", i, j, B[INDEX(i, j, N)], C[INDEX(i, j, N)]);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = 50;  
    if (argc > 1) {
        sscanf(argv[1], "%d", &N);  
    }

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
        print_matrix("A", A, N);
        comp(A, B2, N);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int ctn = 0;
    
    for (int i = 0; i < N - 2; i++) {
        if (id_procs == num_1) {
            int dest = i % num_1;
            int tag = i / num_procs;
            MPI_Send(&A[INDEX(i, 0, N)], N * 3, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }
    }

    
    for (int i = 0; i < (N - 2) / num_procs; i++) {
        if (id_procs != num_procs - 1) {
            MPI_Recv(&A[INDEX(3 * ctn, 0, N)], 3 * N, MPI_DOUBLE, num_procs - 1, ctn, MPI_COMM_WORLD, &status);
            ctn++;
        }
    }
    if (id_procs < (N - 2) % num_procs) {
        MPI_Recv(&A[INDEX(ctn * 3, 0, N)], 3 * N, MPI_DOUBLE, num_procs - 1, ctn, MPI_COMM_WORLD, &status);
        ctn++;
    }

    
    if (id_procs != num_procs - 1) {
        for (int i = 1; i <= ctn; i++) {
            for (int j = 1; j < N - 1; j++) {
                B[INDEX(i, j, N)] = (A[INDEX(i-1, j, N)] + A[INDEX(i, j+1, N)] + A[INDEX(i+1, j, N)] + A[INDEX(i, j-1, N)]) / 4.0;
            }
        }
    }

    
    for (int i = 0; i < N - 2; i++) {
        if (id_procs == num_procs - 1) {
            int src = i % num_procs;
            MPI_Recv(&B[INDEX(i+1, 1, N)], N - 2, MPI_DOUBLE, src, i / num_procs + N, MPI_COMM_WORLD, &status);
        }
        else {
            for (int j = 0; j < ctn; j++)
                MPI_Send(&B[INDEX(j+1, 1, N)], N - 2, MPI_DOUBLE, num_procs - 1, j + N, MPI_COMM_WORLD);
        }
    }

    
    if (id_procs == num_procs - 1) {
        print_matrix("B", B, N);
        if (check(B, B2, N)) {
            printf("Done. No Error\n");
        } else {
            printf("Error Occurred!\n");
        }
    }

    
    free(A);
    free(B);
    free(B2);

    MPI_Finalize();
    return 0;
}
