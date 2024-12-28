#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm world_comm = MPI_COMM_WORLD;
    int world_rank, world_size;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    
    int groupsize = 4; 
    if (argc > 1) {
        groupsize = atoi(argv[1]);
        if (groupsize <= 0) {
            if (world_rank == 0) {
                fprintf(stderr, "请输入一个正整数\n");
            }
            MPI_Abort(world_comm, 1);
        }
    }

    
    int color = world_rank / groupsize;
    int key = world_rank % groupsize;

    
    MPI_Comm split_comm;
    MPI_Comm_split(world_comm, color, key, &split_comm);

    
    int split_rank, split_size;
    MPI_Comm_rank(split_comm, &split_rank);
    MPI_Comm_size(split_comm, &split_size);

    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    
    printf("全局排名: %d, 分组排名: %d/%d, 组号: %d, 节点名称: %s\n", 
           world_rank, split_rank, split_size, color, processor_name);

    
    MPI_Comm_free(&split_comm);

    MPI_Finalize();
    return 0;
}
