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

    MPI_Comm_free(&node_comm);

    MPI_Finalize();
    return 0;
}
