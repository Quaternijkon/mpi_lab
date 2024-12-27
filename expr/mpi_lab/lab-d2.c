#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROOT_ID 0
#define MAX_PROCS_SIZE 16

int main(int argc, char *argv[]) {
    double start_time, end_time, time;
    int procs_id, procs_size;
    MPI_Status status;
    MPI_Request reqSend, reqRecv;

    
    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &procs_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &procs_id);

    
    int N = 0;
    for (int i = 1; i < argc; ++i) {
        char *pos = strstr(argv[i], "-N=");
        if (pos != NULL) {
            sscanf(pos, "-N=%d", &N);
            break;
        }
    }

    
    if (N == 0) {
        if (procs_id == ROOT_ID) {
            printf("未指定矩阵大小 N。请使用 -N=<number> 参数。\n");
        }
        MPI_Finalize();
        return 0;
    }

    
    int procs_size_sqrt = (int)floor(sqrt((double)procs_size));
    if (procs_size_sqrt * procs_size_sqrt != procs_size) {
        if (procs_id == ROOT_ID) {
            printf("处理器数量必须是完全平方数。\n");
        }
        MPI_Finalize();
        return 0;
    }

    
    if (N % procs_size_sqrt != 0) {
        if (procs_id == ROOT_ID) {
            printf("矩阵大小 N 必须能被 sqrt(p) 整除。\n");
        }
        MPI_Finalize();
        return 0;
    }

    
    int n = N / procs_size_sqrt;
    int n_sqr = n * n;

    
    if (procs_size < 4 || procs_size > MAX_PROCS_SIZE) {
        if (procs_id == ROOT_ID) {
            printf("Fox 算法需要至少 4 个处理器，最多 %d 个处理器。\n", MAX_PROCS_SIZE);
        }
        MPI_Finalize();
        return 0;
    }

    
    int *A = (int *)malloc(n_sqr * sizeof(int));
    int *B = (int *)malloc(n_sqr * sizeof(int));
    int *C = (int *)malloc(n_sqr * sizeof(int));
    int *T = (int *)malloc(n_sqr * sizeof(int));

    if (A == NULL || B == NULL || C == NULL || T == NULL) {
        printf("进程 %d 无法分配内存。\n", procs_id);
        MPI_Finalize();
        return -1;
    }

    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = (i + j) * procs_id;
            B[i * n + j] = (i - j) * procs_id;
            C[i * n + j] = 0;
        }
    }

    

    printf("A 在处理器 %d 上:\n", procs_id);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%5d", A[i * n + j]);
        }
        printf("\n");
    }

    printf("B 在处理器 %d 上:\n", procs_id);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%5d", B[i * n + j]);
        }
        printf("\n");
    }


    
    MPI_Comm cart_all, cart_row, cart_col;
    int dims[2], periods[2];
    int procs_cart_rank, procs_coords[2];
    dims[0] = dims[1] = procs_size_sqrt;
    periods[0] = periods[1] = 1; 
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_all);
    MPI_Comm_rank(cart_all, &procs_cart_rank);
    MPI_Cart_coords(cart_all, procs_cart_rank, 2, procs_coords);

    
    MPI_Comm_split(cart_all, procs_coords[0], procs_coords[1], &cart_row);
    MPI_Comm_split(cart_all, procs_coords[1], procs_coords[0], &cart_col);
    int rank_cart_row, rank_cart_col;
    MPI_Comm_rank(cart_row, &rank_cart_row);
    MPI_Comm_rank(cart_col, &rank_cart_col);

    
    for (int round = 0; round < procs_size_sqrt; ++round) {
        
        int send_to = (procs_coords[0] - 1 + procs_size_sqrt) % procs_size_sqrt;
        MPI_Isend(B, n_sqr, MPI_INT, send_to, 1, cart_col, &reqSend);

        
        int broader = (round + procs_coords[0]) % procs_size_sqrt;

        if (broader == procs_coords[1]) {
            
            memcpy(T, A, n_sqr * sizeof(int));
        }

        
        MPI_Bcast(T, n_sqr, MPI_INT, broader, cart_row);

        
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                for (int k = 0; k < n; ++k) {
                    C[row * n + col] += T[row * n + k] * B[k * n + col];
                }
            }
        }

        
        MPI_Wait(&reqSend, &status);

        
        int recv_from = (procs_coords[0] + 1) % procs_size_sqrt;
        MPI_Recv(T, n_sqr, MPI_INT, recv_from, 1, cart_col, &status);
        memcpy(B, T, n_sqr * sizeof(int));
    }

    
    int *C_final = NULL;
    if (procs_id == ROOT_ID) {
        C_final = (int *)malloc(N * N * sizeof(int));
        if (C_final == NULL) {
            printf("根进程无法分配内存用于最终矩阵 C。\n");
            MPI_Finalize();
            return -1;
        }
        
        memset(C_final, 0, N * N * sizeof(int));
    }

    
    MPI_Gather(C, n_sqr, MPI_INT, C_final, n_sqr, MPI_INT, ROOT_ID, cart_all);

    
    if (procs_id == ROOT_ID) {
        printf("最终结果矩阵 C:\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                
                int proc_row = i / n;
                int proc_col = j / n;
                int proc_rank = proc_row * procs_size_sqrt + proc_col;
                
                int value = C_final[proc_rank * n_sqr + (i % n) * n + (j % n)];
                printf("%5d", value);
            }
            printf("\n");
        }
        free(C_final);
    }

    
    if (procs_id == ROOT_ID) {
        end_time = MPI_Wtime();
        printf("任务 %d 耗时 %lf 秒\n", procs_id, end_time - start_time);
    }

    
    MPI_Comm_free(&cart_col);
    MPI_Comm_free(&cart_row);
    MPI_Comm_free(&cart_all);
    free(A);
    free(B);
    free(C);
    free(T);

    MPI_Finalize();
    return 0;
}
