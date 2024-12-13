#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int *sendbuf, *recvbuf;
    int i;
    double start, end, custom_time, alltoall_time;
    FILE *fp;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int *)malloc(size * sizeof(int));
    recvbuf = (int *)malloc(size * sizeof(int));

    for(i = 0; i < size; i++) {
        sendbuf[i] = rank * size + i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for(i = 0; i < size; i++) {
        if(i != rank) {
            MPI_Send(&sendbuf[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Recv(&recvbuf[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            recvbuf[i] = sendbuf[i];
        }
    }
    end = MPI_Wtime();
    custom_time = end - start;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);
    end = MPI_Wtime();
    alltoall_time = end - start;

    double total_custom_time, total_alltoall_time;
    MPI_Reduce(&custom_time, &total_custom_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alltoall_time, &total_alltoall_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
