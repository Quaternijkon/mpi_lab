#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define IDX(i, j, n) ((i)*(n) + (j))


void initialize_matrix(double *mat, int n) {
    for(int i = 0; i < n*n; i++) {
        mat[i] = rand() % 10; 
    }
}


void print_matrix(double *mat, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%6.2f ", mat[IDX(i,j,n)]);
        }
        printf("\n");
    }
}


void matmul_add(double *A, double *B, double *C, int n) {
    for(int i = 0; i < n; i++) {
        for(int k = 0; k < n; k++) {
            for(int j = 0; j < n; j++) {
                C[IDX(i,j,n)] += A[IDX(i,k,n)] * B[IDX(k,j,n)];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
    int q = (int)sqrt((double)size);
    if(q * q != size) {
        if(rank == 0) {
            printf("进程数 p 必须是完全平方数。\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    
    int n = 4; 
    if(argc > 1) {
        n = atoi(argv[1]);
    }
    
    if(n % q != 0) {
        if(rank == 0) {
            printf("矩阵大小 n 必须能被 sqrt(p) 整除。\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    int block_size = n / q;
    
    
    int dims[2] = {q, q};
    int periods[2] = {1, 1}; 
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];
    
    
    MPI_Comm row_comm;
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);
    
    
    MPI_Comm col_comm;
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);
    
    
    double *A_block = (double*)malloc(block_size * block_size * sizeof(double));
    double *B_block = (double*)malloc(block_size * block_size * sizeof(double));
    double *C_block = (double*)malloc(block_size * block_size * sizeof(double));
    
    
    for(int i = 0; i < block_size * block_size; i++) {
        C_block[i] = 0.0;
    }
    
    
    double *A = NULL;
    double *B = NULL;
    if(rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        initialize_matrix(A, n);
        initialize_matrix(B, n);
        printf("Matrix A:\n");
        print_matrix(A, n);
        printf("Matrix B:\n");
        print_matrix(B, n);
    }
    
    
    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, n, MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, block_size * sizeof(double), &block_type);
    MPI_Type_commit(&block_type);
    
    
    double *A_sub = (double*)malloc(block_size * block_size * sizeof(double));
    double *B_sub = (double*)malloc(block_size * block_size * sizeof(double));
    
    if(rank == 0) {
        
        
    }
    
    
    int *sendcounts = NULL;
    int *displs = NULL;
    if(rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            sendcounts[i] = 1;
            int row = i / q;
            int col = i % q;
            displs[i] = row * n * block_size + col * block_size;
        }
    }
    
    
    MPI_Scatterv(A, sendcounts, displs, block_type, A_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, block_type, B_block, block_size * block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        free(sendcounts);
        free(displs);
    }
    
    
    if(rank == 0) {
        free(A);
        free(B);
    }
    
    
    for(int stage = 0; stage < q; stage++) {
        int root = (my_row + stage) % q;
        double *A_broadcast = (double*)malloc(block_size * block_size * sizeof(double));
        
        if(root == my_col) {
            
            for(int i = 0; i < block_size * block_size; i++) {
                A_broadcast[i] = A_block[i];
            }
        }
        
        
        MPI_Bcast(A_broadcast, block_size * block_size, MPI_DOUBLE, root, row_comm);
        
        
        matmul_add(A_broadcast, B_block, C_block, block_size);
        
        free(A_broadcast);
        
        
        
        int src, dest;
        MPI_Cart_shift(grid_comm, 0, -1, &src, &dest);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE, dest, 0, src, 0, grid_comm, MPI_STATUS_IGNORE);
    }
    
    
    double *C = NULL;
    if(rank == 0) {
        C = (double*)malloc(n * n * sizeof(double));
    }
    
    
    if(rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i++) {
            sendcounts[i] = 1;
            int row = i / q;
            int col = i % q;
            displs[i] = row * n * block_size + col * block_size;
        }
    }
    
    
    
    
    MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE, C, sendcounts, displs, block_type, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        printf("Matrix C = A * B:\n");
        print_matrix(C, n);
        free(C);
        free(sendcounts);
        free(displs);
    }
    
    
    free(A_block);
    free(B_block);
    free(C_block);
    free(A_sub);
    free(B_sub);
    
    MPI_Type_free(&block_type);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}
