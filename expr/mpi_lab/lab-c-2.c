#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>  

int main(int argc, char *argv[])
{
    int id_procs, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id_procs);

    
    int temp_size = num_procs;
    while (temp_size > 1) {
        if (temp_size % 2 != 0) {
            if (id_procs == 0) {
                printf("进程数必须是2的幂。\n");
            }
            MPI_Finalize();
            return 0;
        }
        temp_size /= 2;
    }

    
    int log2_size = (int)log2((double)num_procs);

    
    srand(time(NULL) + id_procs);
    int data = rand() % 100;  
    int recvdata = 0;
    MPI_Status status;

    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    
    for(int i = 1; i <= log2_size; i++) {
        int tag = 1 << (i-1);
        int partner = id_procs ^ tag;  
        if (id_procs < partner) {
            
            MPI_Send(&data, 1, MPI_INT, partner, tag, MPI_COMM_WORLD);
        }
        else {
            
            MPI_Recv(&recvdata, 1, MPI_INT, partner, tag, MPI_COMM_WORLD, &status);
            data += recvdata;
        }
    }

    
    for(int i = log2_size; i >=1; i--) {
        int tag = 1 << (i-1);
        int partner = id_procs ^ tag;  
        if (id_procs < partner) {
            
            MPI_Recv(&recvdata, 1, MPI_INT, partner, tag, MPI_COMM_WORLD, &status);
            data = recvdata;  
        }
        else {
            
            MPI_Send(&data, 1, MPI_INT, partner, tag, MPI_COMM_WORLD);
        }
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    
    if (id_procs == 0) {
        printf("最终的全局总和为 %d\n", data);
        printf("所有进程的总用时: %f 秒\n", elapsed_time);
    }

    
    printf("进程 %d 的全局总和为 %d\n", id_procs, data);

    MPI_Finalize();
    return 0;
}
