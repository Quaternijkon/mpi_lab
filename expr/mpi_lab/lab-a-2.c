#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm world_comm = MPI_COMM_WORLD;
    int world_rank, world_size;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(world_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("全局排名: %d, 节点内排名: %d/%d, 节点名称: %s\n", 
           world_rank, node_rank, node_size, processor_name);

    MPI_Comm node_roots_comm;
    int color = (node_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world_comm, color, world_rank, &node_roots_comm);

    char message[256];
    if (node_rank == 0) {
        if (world_rank == 0) {
            strcpy(message, "Hello from the global root process!");
        }
    }

    if (color == 0) {
        MPI_Bcast(message, sizeof(message), MPI_CHAR, 
                  (world_rank == 0) ? 0 : MPI_UNDEFINED, node_roots_comm);
    }

    MPI_Barrier(world_comm);

    MPI_Bcast(message, sizeof(message), MPI_CHAR, 0, node_comm);

    printf("全局排名: %d, 节点内排名: %d/%d, 接收到的消息: %s\n", 
           world_rank, node_rank, node_size, message);

    if (color == 0) {
        MPI_Comm_free(&node_roots_comm);
    }
    MPI_Comm_free(&node_comm);

    MPI_Finalize();
    return 0;
}
