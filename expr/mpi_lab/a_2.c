#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define ROOT 0
#define MESSAGE "Hello from root!"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
 
    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    MPI_Comm leader_comm;
    int is_leader = (node_rank == 0) ? 1 : 0;
    MPI_Comm_split(MPI_COMM_WORLD, is_leader, world_rank, &leader_comm);

    char message_buffer[100];
    memset(message_buffer, 0, sizeof(message_buffer));

    if (leader_comm != MPI_COMM_NULL) {
        int leader_rank, leader_size;
        MPI_Comm_rank(leader_comm, &leader_rank);
        MPI_Comm_size(leader_comm, &leader_size);

        if (world_rank == ROOT) {
            strcpy(message_buffer, MESSAGE);
        }

        MPI_Bcast(message_buffer, sizeof(message_buffer), MPI_CHAR, 
                  (world_rank == ROOT) ? 0 : MPI_UNDEFINED, leader_comm);
    }

    MPI_Bcast(message_buffer, sizeof(message_buffer), MPI_CHAR, 0, node_comm);

    printf("World Rank: %d received message: %s\n", world_rank, message_buffer);

    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
    }
    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
