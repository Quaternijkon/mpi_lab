#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROOT 0
#define TAG 100

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int send_count = 1000; 
    int recv_count = 1000;

    int *send_buffer = (int *)malloc(send_count * world_size * sizeof(int));
    int *recv_buffer = (int *)malloc(recv_count * world_size * sizeof(int));

    for(int i = 0; i < send_count * world_size; i++) {
        send_buffer[i] = world_rank;
    }

    double start_time = MPI_Wtime();

    for(int i = 0; i < world_size; i++) {
        if(i == world_rank) {
            memcpy(&recv_buffer[i * recv_count], &send_buffer[i * send_count], send_count * sizeof(int));
        } else {
            MPI_Send(&send_buffer[i * send_count], send_count, MPI_INT, i, TAG, MPI_COMM_WORLD);
            MPI_Recv(&recv_buffer[i * recv_count], recv_count, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    double end_time = MPI_Wtime();
    double simulation_time = end_time - start_time;

    printf("Process %d: Fake MPI_Alltoall Time = %f seconds\n", world_rank, simulation_time);

    free(send_buffer);
    free(recv_buffer);

    MPI_Barrier(MPI_COMM_WORLD);

    send_buffer = (int *)malloc(send_count * world_size * sizeof(int));
    recv_buffer = (int *)malloc(recv_count * world_size * sizeof(int));

    for(int i = 0; i < send_count * world_size; i++) {
        send_buffer[i] = world_rank;
    }

    start_time = MPI_Wtime();

    MPI_Alltoall(send_buffer, send_count, MPI_INT, recv_buffer, recv_count, MPI_INT, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    double alltoall_time = end_time - start_time;

    printf("Process %d: MPI_Alltoall Time = %f seconds\n", world_rank, alltoall_time);

    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return 0;
}
