#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int is_power_of_two(int n) {
    return (n != 0) && ((n & (n -1)) == 0);
}

int compute_log2(int n) {
    int logn = 0;
    while (n >>= 1) ++logn;
    return logn;
}

int main(int argc, char** argv) {
    int rank, size;
    int logN;
    double local_data, local_sum, recv_sum;
    MPI_Status status;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    if (!is_power_of_two(size)) {
        if (rank == 0) {
            fprintf(stderr, "进程数必须是2的幂次方。\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    logN = compute_log2(size);

    
    local_data = (double)(rank + 1);
    local_sum = local_data;

    
    for (int i = 0; i < logN; i++) {
        int partner = rank ^ (1 << i); 

        
        MPI_Sendrecv(&local_sum, 1, MPI_DOUBLE, partner, 0,
                     &recv_sum, 1, MPI_DOUBLE, partner, 0,
                     MPI_COMM_WORLD, &status);

        
        local_sum += recv_sum;
    }

    
    printf("进程 %d 的全和结果: %f\n", rank, local_sum);

    
    MPI_Finalize();
    return 0;
}
