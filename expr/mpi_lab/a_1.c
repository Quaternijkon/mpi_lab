#include <mpi.h>
#include <stdio.h>
#include <string.h>

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

    int node_id;
    MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &node_comm);
    MPI_Comm_rank(node_comm, &node_id);

    if (world_rank == 0) {
        printf("Total processes: %d\n", world_size);
    }

    printf("World Rank: %d, Node Rank: %d, Node Size: %d\n", world_rank, node_rank, node_size);

    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return 0;
}
