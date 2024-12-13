#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    double local_value, sum;
    int log2_size;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    log2_size = (int)(log((double)size) / log(2.0));
    if (pow(2, log2_size) != size) {
        if (rank == 0) {
            printf("进程数必须是2的幂。\n");
        }
        MPI_Finalize();
        return 0;
    }

    
    local_value = (double)(rank + 1);
    sum = local_value;

    
    for (i = 0; i < log2_size; i++) {
        int partner = rank ^ (1 << i);
        double recv_val;

        if (rank < partner) {
            
            MPI_Send(&sum, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
        } else {
            
            MPI_Recv(&recv_val, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += recv_val;
        }
        MPI_Barrier(MPI_COMM_WORLD); 
    }

    
    for (i = 0; i < log2_size; i++) {
        int partner = rank ^ (1 << i);
        if (rank < partner) {
            
            MPI_Send(&sum, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
        } else {
            
            MPI_Recv(&sum, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD); 
    }

    
    printf("进程 %d 的全局总和为 %.2f\n", rank, sum);

    MPI_Finalize();
    return 0;
}
