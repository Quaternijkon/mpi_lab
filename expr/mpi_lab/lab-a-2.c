#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ROOT 0
#define MESSAGE "Hello from root!"
#define SEQ_SIZE 16  

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    
    int groupsize = 4; 
    if (argc > 1) {
        groupsize = atoi(argv[1]);
        if (groupsize <= 0) {
            if (world_rank == ROOT) {
                fprintf(stderr, "Invalid groupsize. It must be a positive integer.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    
    int color = world_rank / groupsize;
    int key = world_rank % groupsize;

    
    MPI_Comm splitWorld;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &splitWorld);

    int split_rank, split_size;
    MPI_Comm_rank(splitWorld, &split_rank);
    MPI_Comm_size(splitWorld, &split_size);

    
    if (split_rank >= split_size) {
        
        MPI_Comm_free(&splitWorld);
        MPI_Finalize();
        return 0;
    }

    
    int tag = 0;
    MPI_Status status;
    char seq[SEQ_SIZE] = "SampleSeqData"; 
    char seqin[SEQ_SIZE];
    memset(seqin, 0, SEQ_SIZE);

    
    if (split_rank == 0) {
        
        strcpy(seqin, seq);
        for(int i = 1; i < split_size; i++) {
            MPI_Send(seq, SEQ_SIZE, MPI_CHAR, i, tag, splitWorld);
        }
    } else {
        
        MPI_Recv(seqin, SEQ_SIZE, MPI_CHAR, 0, tag, splitWorld, &status);
    }

    
    MPI_Bcast(seqin, SEQ_SIZE, MPI_CHAR, 0, splitWorld);

    
    printf("全局排名: %d, 分组排名: %d/%d, 收到信息: %s\n", 
           world_rank, split_rank, split_size, seqin);

    
    MPI_Comm_free(&splitWorld);
    MPI_Finalize();
    return 0;
}
