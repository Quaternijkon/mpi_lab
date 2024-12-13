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
    if (node_rank == 0) {
        node_id = world_rank; 
    }
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm);

    int total_nodes;
    MPI_Comm node_leader_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_rank == 0, node_id, &node_leader_comm);

    if (node_rank == 0) {
        MPI_Comm_size(node_leader_comm, &total_nodes);
    }
    MPI_Bcast(&total_nodes, 1, MPI_INT, 0, node_leader_comm);

    if (node_rank == 0) {
        printf("Node %d has %d processes\n", node_id, node_size);
    }

    MPI_Comm_free(&node_comm);
    MPI_Comm_free(&node_leader_comm);
    MPI_Finalize();
    return 0;
}
