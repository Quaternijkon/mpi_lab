#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


int My_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               MPI_Comm comm) {
    int rank, size, i;
    MPI_Status status;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int send_type_size, recv_type_size;
    MPI_Type_size(sendtype, &send_type_size);
    MPI_Type_size(recvtype, &recv_type_size);

    
    for (i = 0; i < size; i++) {
        if (i == rank) {
            
            memcpy((char *)recvbuf + i * recvcount * recv_type_size,
                   (char *)sendbuf + i * sendcount * send_type_size,
                   sendcount * send_type_size);
        } else {
            
            MPI_Send((char *)sendbuf + i * sendcount * send_type_size,
                     sendcount, sendtype, i, 0, comm);
            
            MPI_Recv((char *)recvbuf + i * recvcount * recv_type_size,
                     recvcount, recvtype, i, 0, comm, &status);
        }
    }

    return MPI_SUCCESS;
}

int main(int argc, char *argv[]) {
    int rank, size;
    const int num_iterations = 100; 
    double start_time, end_time;
    double my_alltoall_time = 0.0, mpi_alltoall_time = 0.0;
    int i;
    int message_size = 1024; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    char *sendbuf = (char *)malloc(size * message_size * sizeof(char));
    char *recvbuf_my = (char *)malloc(size * message_size * sizeof(char));
    char *recvbuf_mpi = (char *)malloc(size * message_size * sizeof(char));

    
    for (i = 0; i < size * message_size; i++) {
        sendbuf[i] = 'a' + (rank % 26);
    }

    
    for (i = 0; i < num_iterations; i++) {
        memset(recvbuf_my, 0, size * message_size);
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        My_Alltoall(sendbuf, message_size, MPI_CHAR,
                    recvbuf_my, message_size, MPI_CHAR, MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        my_alltoall_time += (end_time - start_time);
    }

    
    for (i = 0; i < num_iterations; i++) {
        memset(recvbuf_mpi, 0, size * message_size);
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        MPI_Alltoall(sendbuf, message_size, MPI_CHAR,
                     recvbuf_mpi, message_size, MPI_CHAR, MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        mpi_alltoall_time += (end_time - start_time);
    }

    
    double avg_my_alltoall = my_alltoall_time / num_iterations;
    double avg_mpi_alltoall = mpi_alltoall_time / num_iterations;

    
    double max_my_alltoall, max_mpi_alltoall;
    MPI_Reduce(&avg_my_alltoall, &max_my_alltoall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_mpi_alltoall, &max_mpi_alltoall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    if (rank == 0) {
        FILE *fp = fopen("mpi_alltoall_results.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Error opening file for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(fp, "MPI_Alltoall Performance Comparison\n");
        fprintf(fp, "Number of Processes: %d\n", size);
        fprintf(fp, "Message Size: %d bytes per send\n", message_size);
        fprintf(fp, "Iterations: %d\n\n", num_iterations);
        fprintf(fp, "Custom My_Alltoall Average Time: %f seconds\n", max_my_alltoall);
        fprintf(fp, "MPI_Alltoall Average Time: %f seconds\n", max_mpi_alltoall);
        fprintf(fp, "Speedup (MPI_Alltoall / My_Alltoall): %f\n",
                max_my_alltoall / max_mpi_alltoall);

        fclose(fp);
        printf("Results saved to mpi_alltoall_results.txt\n");
    }

    
    free(sendbuf);
    free(recvbuf_my);
    free(recvbuf_mpi);

    MPI_Finalize();
    return 0;
}
