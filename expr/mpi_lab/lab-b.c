#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 10000  
#define ITERATIONS 100    

int main(int argc, char *argv[]) {
    int rank, size;
    int *sendbuf, *recvbuf;
    int i, iter;
    double start, end, custom_total_time = 0.0, alltoall_total_time = 0.0;
    FILE *fp;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    sendbuf = (int *)malloc(DATA_SIZE * size * sizeof(int));
    recvbuf = (int *)malloc(DATA_SIZE * size * sizeof(int));

    
    for(i = 0; i < DATA_SIZE * size; i++) {
        sendbuf[i] = rank * DATA_SIZE * size + i; 
    }

    
    for(iter = 0; iter < ITERATIONS; iter++) {
        
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        for(i = 0; i < size; i++) {
            if(i != rank) {
                
                MPI_Send(&sendbuf[i * DATA_SIZE], DATA_SIZE, MPI_INT, i, 0, MPI_COMM_WORLD);
                
                MPI_Recv(&recvbuf[i * DATA_SIZE], DATA_SIZE, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                
                for(int j = 0; j < DATA_SIZE; j++) {
                    recvbuf[i * DATA_SIZE + j] = sendbuf[i * DATA_SIZE + j];
                }
            }
        }
        end = MPI_Wtime();
        custom_total_time += (end - start);
    }

    double custom_avg_time = custom_total_time / ITERATIONS;

    
    for(iter = 0; iter < ITERATIONS; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        MPI_Alltoall(sendbuf, DATA_SIZE, MPI_INT, recvbuf, DATA_SIZE, MPI_INT, MPI_COMM_WORLD);
        end = MPI_Wtime();
        alltoall_total_time += (end - start);
    }

    double alltoall_avg_time = alltoall_total_time / ITERATIONS;

    
    double total_custom_time, total_alltoall_time;
    MPI_Reduce(&custom_avg_time, &total_custom_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alltoall_avg_time, &total_alltoall_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        
        fp = fopen("performance_data.csv", "a");
        if(fp == NULL) {
            fprintf(stderr, "Error opening file!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "%d,%f,%f\n", size, total_custom_time, total_alltoall_time);
        fclose(fp);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
