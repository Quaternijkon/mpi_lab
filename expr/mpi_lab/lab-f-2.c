#include <mpi.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int N = 7;


#define LOCAL_INDEX(i, j, cols) (((i)*(cols)) + (j))


static inline int GLOBAL_INDEX(int i, int j) {
    return i * N + j;
}


void initialize_matrix(double *a, int num_rows, int num_cols) {
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            a[i * num_cols + j] = (double)(i * num_cols + j); 
        }
    }
}


void comp(double *A, double *B, int a, int b, int local_cols) {
    for(int i = 1; i <= a; i++) {
        for(int j = 1; j <= b; j++) {
            B[LOCAL_INDEX(i, j, local_cols)] = (A[LOCAL_INDEX(i-1, j, local_cols)] +
                                               A[LOCAL_INDEX(i, j+1, local_cols)] +
                                               A[LOCAL_INDEX(i+1, j, local_cols)] +
                                               A[LOCAL_INDEX(i, j-1, local_cols)]) / 4.0;
        }
    }
}


int check(double *B, double *C) {
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {
            if (fabs(B[GLOBAL_INDEX(i, j)] - C[GLOBAL_INDEX(i, j)]) >= 1e-2) {
                printf("B[%d,%d] = %lf not %lf!\n", i, j, B[GLOBAL_INDEX(i, j)], C[GLOBAL_INDEX(i, j)]);
                return 0;
            }
        }
    }
    return 1;
}


void print_matrix(double *matrix, int num_rows, int num_cols, const char *name) {
    printf("Matrix %s:\n", name);
    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            printf("%6.2lf ", matrix[i * num_cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    double *A, *B, *B2;
    
    int id_procs, num_procs;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);

    
    if (id_procs == 0) {
        if (argc > 1) {
            int input_N = atoi(argv[1]);
            if(input_N > 0){
                N = input_N;
            }
            else{
                printf("Invalid N value provided. Using default N = 7.\n");
            }
        }
    }

    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    int rows = (int)sqrt(num_procs);
    int cols = num_procs / rows;
    if(rows * cols != num_procs) {
        if(id_procs == 0)
            printf("Number of processes must be a perfect square.\n");
        MPI_Finalize();
        return 0;
    }

    int a = (N-2 + rows-1) / rows; 
    int b = (N-2 + cols-1) / cols; 
    int local_rows = a + 2; 
    int local_cols = b + 2; 

    A = (double*)malloc(local_rows * local_cols * sizeof(double));
    B = (double*)malloc(local_rows * local_cols * sizeof(double));
    B2= (double*)malloc(local_rows * local_cols * sizeof(double));

    
    for(int i = 0; i < local_rows * local_cols; i++) {
        A[i] = 0.0;
        B[i] = 0.0;
        B2[i] = 0.0;
    }

    
    if (id_procs == 0) {
        double *full_A = (double*)malloc(N*N*sizeof(double));
        double *full_B2 = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        
        
        
        double *temp_A = (double*)malloc(N*N*sizeof(double));
        double *temp_B = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(temp_A, N, N);
        
        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                temp_B[GLOBAL_INDEX(i,j)] = (temp_A[GLOBAL_INDEX(i-1,j)] +
                                             temp_A[GLOBAL_INDEX(i,j+1)] +
                                             temp_A[GLOBAL_INDEX(i+1,j)] +
                                             temp_A[GLOBAL_INDEX(i,j-1)]) / 4.0;
            }
        }
        
        for(int i = 0; i < N*N; i++) {
            full_B2[i] = temp_B[i];
        }
        free(temp_A);
        free(temp_B);
        
        for(int p = 0; p < num_procs; p++) {
            int proc_row = p / cols;
            int proc_col = p % cols;
            
            double *send_buffer = (double*)malloc(local_rows * local_cols * sizeof(double));
            
            for(int i = 0; i < local_rows; i++) {
                for(int j = 0; j < local_cols; j++) {
                    int global_i = proc_row * a + (i - 1);
                    int global_j = proc_col * b + (j - 1);
                    if(global_i < 0 || global_i >= N || global_j < 0 || global_j >= N)
                        send_buffer[i * local_cols + j] = 0.0; 
                    else
                        send_buffer[i * local_cols + j] = full_A[GLOBAL_INDEX(global_i, global_j)];
                }
            }
            if(p == 0) {
                
                for(int i = 0; i < local_rows * local_cols; i++) {
                    A[i] = send_buffer[i];
                }
            }
            else {
                
                MPI_Send(send_buffer, local_rows * local_cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
            free(send_buffer);
        }
        
        free(full_A);
        free(full_B2);
    }
    else {
        
        MPI_Recv(A, local_rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    
    MPI_Barrier(MPI_COMM_WORLD);

    
    double start_time = MPI_Wtime();

    
    comp(A, B, a, b, local_cols);

    
    
    int proc_row = id_procs / cols;
    int proc_col = id_procs % cols;
    
    int start_i = proc_row * a;
    int start_j = proc_col * b;
    
    for(int i = 1; i <= a; i++) {
        for(int j = 1; j <= b; j++) {
            int global_i = start_i + (i - 1);
            int global_j = start_j + (j - 1);
            
            if(global_i == 0 || global_j == 0 || global_i == N-1 || global_j == N-1) {
                B[LOCAL_INDEX(i, j, local_cols)] = 0.0;
            }
        }
    }

    
    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed;

    
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    if (id_procs == 0) {
        double *full_B = (double*)malloc(N*N*sizeof(double));
        
        for(int i = 0; i < N*N; i++) {
            full_B[i] = 0.0;
        }
        
        for(int i = 1; i <= a; i++) {
            for(int j = 1; j <= b; j++) {
                int global_i = 0 * a + (i - 1);
                int global_j = 0 * b + (j - 1);
                if(global_i < N && global_j < N)
                    full_B[GLOBAL_INDEX(global_i, global_j)] = B[LOCAL_INDEX(i, j, local_cols)];
            }
        }
        
        for(int p = 1; p < num_procs; p++) {
            double *recv_buffer = (double*)malloc(local_rows * local_cols * sizeof(double));
            MPI_Recv(recv_buffer, local_rows * local_cols, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            int proc_row_p = p / cols;
            int proc_col_p = p % cols;
            for(int i = 1; i <= a; i++) {
                for(int j = 1; j <= b; j++) {
                    int global_i = proc_row_p * a + (i - 1);
                    int global_j = proc_col_p * b + (j - 1);
                    if(global_i < N && global_j < N)
                        full_B[GLOBAL_INDEX(global_i, global_j)] = recv_buffer[LOCAL_INDEX(i, j, local_cols)];
                }
            }
            free(recv_buffer);
        }

        
        double *full_A = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        print_matrix(full_A, N, N, "A");
        print_matrix(full_B, N, N, "B");

        
        printf("Computation Time: %lf seconds\n", max_elapsed);

        
        
        double *computed_B2 = (double*)malloc(N*N*sizeof(double));
        initialize_matrix(full_A, N, N);
        
        for(int i = 1; i < N-1; i++) {
            for(int j = 1; j < N-1; j++) {
                computed_B2[GLOBAL_INDEX(i,j)] = (full_A[GLOBAL_INDEX(i-1,j)] +
                                                 full_A[GLOBAL_INDEX(i,j+1)] +
                                                 full_A[GLOBAL_INDEX(i+1,j)] +
                                                 full_A[GLOBAL_INDEX(i,j-1)]) / 4.0;
            }
        }
        if (check(full_B, computed_B2)) {
            printf("Done. No Error\n");
        } else {
            printf("Error!\n");
        }

        free(full_A);
        free(full_B);
        free(computed_B2);
    }
    else {
        
        MPI_Send(B, local_rows * local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    
    

    free(A);
    free(B);
    free(B2);
    MPI_Finalize();
    return 0;
}
