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

    int global_root = 0;

    MPI_Comm group_roots_comm;
    int is_group_root = (node_rank == 0) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(world_comm, is_group_root, world_rank, &group_roots_comm);

    char message[100];
    memset(message, 0, sizeof(message));

    if (is_group_root) {
        int group_roots_rank, group_roots_size;
        MPI_Comm_rank(group_roots_comm, &group_roots_rank);
        MPI_Comm_size(group_roots_comm, &group_roots_size);

        if (world_rank == global_root) {
            strcpy(message, "Hello from global root");
        }

        MPI_Bcast(message, sizeof(message), MPI_CHAR, 0, group_roots_comm);

        MPI_Bcast(message, sizeof(message), MPI_CHAR, 0, node_comm);
    } else {
        MPI_Bcast(message, sizeof(message), MPI_CHAR, 0, node_comm);
    }

    printf("全局排名: %d, 节点内排名: %d/%d, 节点名称: %s, 接收到消息: %s\n", 
           world_rank, node_rank, node_size, processor_name, message);

    if (is_group_root) {
        MPI_Comm_free(&group_roots_comm);
    }
    MPI_Comm_free(&node_comm);

    MPI_Finalize();
    return 0;
}
