#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * 检查进程数是否为2的幂次方
 */
int is_power_of_two(int n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}

int main(int argc, char *argv[]) {
    int rank, size;
    double local_value; 
    double total_sum = 0.0; 
    int log_p; 
    int step;
    int partner;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (!is_power_of_two(size)) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes must be a power of 2.\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    log_p = (int)(log((double)size) / log(2.0));
    if (pow(2, log_p) != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes must be a power of 2.\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    local_value = (double)rank;

    
    printf("Process %d initial value: %f\n", rank, local_value);

    
    for (step = 0; step < log_p; step++) {
        
        partner = rank ^ (1 << step);

        double recv_value;

        
        MPI_Sendrecv(&local_value, 1, MPI_DOUBLE, partner, 0,
                     &recv_value, 1, MPI_DOUBLE, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        
        local_value += recv_value;

        
        printf("Process %d after step %d has value: %f\n", rank, step, local_value);
    }

    
    total_sum = local_value;

    
    printf("Process %d total sum: %f\n", rank, total_sum);

    MPI_Finalize();
    return 0;
}
