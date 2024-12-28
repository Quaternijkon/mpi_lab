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

    
    if (argc >= 2) {
        send_count = atoi(argv[1]);
        if (send_count <= 0) {
            if (world_rank == ROOT) {
                printf("Invalid send_count provided. Using default value 1000.\n");
            }
            send_count = 1000;
        }
    }

    if (argc >= 3) {
        recv_count = atoi(argv[2]);
        if (recv_count <= 0) {
            if (world_rank == ROOT) {
                printf("Invalid recv_count provided. Using default value 1000.\n");
            }
            recv_count = 1000;
        }
    }

    
    if (world_rank == ROOT) {
        printf("Configuration:\n");
        printf("  send_count = %d\n", send_count);
        printf("  recv_count = %d\n", recv_count);
        printf("  world_size = %d\n\n", world_size);
    }

    
    int *send_buffer = (int *)malloc(send_count * world_size * sizeof(int));
    int *recv_buffer = (int *)malloc(recv_count * world_size * sizeof(int));

    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    
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

    
    double total_simulation_time;
    MPI_Reduce(&simulation_time, &total_simulation_time, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    
    MPI_Barrier(MPI_COMM_WORLD);

    
    send_buffer = (int *)malloc(send_count * world_size * sizeof(int));
    recv_buffer = (int *)malloc(recv_count * world_size * sizeof(int));

    if (send_buffer == NULL || recv_buffer == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    
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

    
    double total_alltoall_time;
    MPI_Reduce(&alltoall_time, &total_alltoall_time, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

    
    if(world_rank == ROOT) {
        double average_simulation_time = total_simulation_time / world_size;
        double average_alltoall_time = total_alltoall_time / world_size;

        printf("\nAverage Fake MPI_Alltoall Time = %f seconds\n", average_simulation_time);
        printf("Average MPI_Alltoall Time = %f seconds\n", average_alltoall_time);
    }

    MPI_Finalize();
    return 0;
}
