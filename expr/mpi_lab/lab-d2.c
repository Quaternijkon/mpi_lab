#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>

void matrixMultiply(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int n) {
    // Perform matrix multiplication: C = A * B + C
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int grid_size = static_cast<int>(std::sqrt(size)); // Process grid dimensions
    if (grid_size * grid_size != size) {
        if (rank == 0) std::cerr << "Number of processes must be a perfect square.\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int n = 4; // Matrix dimension (example, should be divisible by grid_size)
    int block_size = n / grid_size; // Size of sub-matrices

    // Create a 2D Cartesian grid
    MPI_Comm grid_comm;
    int dims[2] = {grid_size, grid_size};
    int periods[2] = {1, 1}; // Allow wrap-around
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    int row = coords[0];
    int col = coords[1];

    // Initialize matrices
    std::vector<int> A(block_size * block_size, rank); // Example initialization
    std::vector<int> B(block_size * block_size, rank);
    std::vector<int> C(block_size * block_size, 0);

    std::vector<int> A_temp(block_size * block_size);
    std::vector<int> B_temp(block_size * block_size);

    for (int k = 0; k < grid_size; ++k) {
        // Broadcast A
        int Bcaster = row * grid_size + (row + k) % grid_size;
        if (rank == Bcaster) {
            A_temp = A;
            for (int l = 0; l < grid_size; ++l) {
                int dst = row * grid_size + l;
                if (dst != rank) {
                    MPI_Send(A_temp.data(), block_size * block_size, MPI_INT, dst, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(A_temp.data(), block_size * block_size, MPI_INT, Bcaster, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute
        matrixMultiply(A_temp, B, C, block_size);

        // Circular shift B upwards
        int send_B = (row * grid_size + col + grid_size - 1) % grid_size;
        int recv_B = (row * grid_size + col + 1) % grid_size;
        if ((row % 2) == 0) {
            MPI_Send(B.data(), block_size * block_size, MPI_INT, send_B, 1, MPI_COMM_WORLD);
            MPI_Recv(B_temp.data(), block_size * block_size, MPI_INT, recv_B, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(B_temp.data(), block_size * block_size, MPI_INT, recv_B, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(B.data(), block_size * block_size, MPI_INT, send_B, 1, MPI_COMM_WORLD);
        }
        B = B_temp;
    }

    // Output the result
    if (rank == 0) {
        std::cout << "Matrix multiplication result:\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            std::cout << "Process " << rank << " result:\n";
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    std::cout << C[j * block_size + k] << " ";
                }
                std::cout << "\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
