#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ITERATIONS 10  

int main(int argc, char *argv[]) {
    int rank, size;
    int P, Q;
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s P Q\n", argv[0]);
            fprintf(stderr, "Where P is the number of parameter servers and Q is the number of workers (N = P + Q).\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    P = atoi(argv[1]);
    Q = atoi(argv[2]);

    
    if (P + Q != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: P + Q must be equal to the number of MPI processes (%d).\n", size);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    
    srand(time(NULL) + rank);

    if (rank < P) {
        
        
        
        
        int num_workers = 0;
        
        int *worker_ranks = (int *)malloc(Q * sizeof(int));
        if (worker_ranks == NULL) {
            fprintf(stderr, "Server %d: Memory allocation failed.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        
        for (int w = 0; w < Q; w++) {
            int worker_rank = P + w;
            if ((w) % P == rank) {
                worker_ranks[num_workers++] = worker_rank;
            }
        }

        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            double sum = 0.0;
            double recv_val;
            
            
            for (int w = 0; w < num_workers; w++) {
                MPI_Status status;
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, worker_ranks[w], 0, MPI_COMM_WORLD, &status);
                sum += recv_val;
            }

            
            double average = sum / num_workers;

            
            for (int w = 0; w < num_workers; w++) {
                MPI_Send(&average, 1, MPI_DOUBLE, worker_ranks[w], 0, MPI_COMM_WORLD);
            }

            
            printf("Server %d, Iteration %d: Computed average = %f\n", rank, iter, average);
        }

        free(worker_ranks);
    }
    else {
        
        
        int worker_index = rank - P;
        int server_rank = P + (worker_index % P); 

        
        server_rank = worker_index % P;

        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            
            double rand_num = ((double)rand()) / RAND_MAX;

            
            MPI_Send(&rand_num, 1, MPI_DOUBLE, server_rank, 0, MPI_COMM_WORLD);

            
            double average;
            MPI_Recv(&average, 1, MPI_DOUBLE, server_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            
            printf("Worker %d, Iteration %d: Sent = %f, Received average = %f\n", rank, iter, rand_num, average);
        }
    }

    
    MPI_Finalize();
    return 0;
}
