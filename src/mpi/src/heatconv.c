#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#ifdef _OPENMP

#include <omp.h>

#endif

#define TRUE 1
#define FALSE 0

#define NX_HEAT 20
#define NY_HEAT 20

#define EXEC_STEPS 10000
#define CONVERGENCE_FREQ_STEPS 100

#define DIMENSIONALITY 2

#define NORTH       0
#define SOUTH       1
#define WEST        2
#define EAST        3

#define SEND        0
#define RECEIVE     1


#define UAT_MODE 1


struct Parms {
    float cx;
    float cy;
} parms = {0.1, 0.1};


int main(int argc, char **argv) {
    void updateInner(), updateOuter();

    int currentRank;
    int commSize;

    double startTime, endTime;

    char processorName[MPI_MAX_PROCESSOR_NAME];
    int processorNameLen;

    int version, subversion;

    float grid[2][NX_HEAT + 2][NY_HEAT + 2];
    int currentGrid = 0;
    int neighbors[4];

    int convergenceCheck = 1;

    MPI_Init(&argc, &argv);

    MPI_Get_version(&version, &subversion);
    MPI_Get_processor_name(processorName, &processorNameLen);

    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);

    MPI_Request request[2][2][4];

    if (currentRank == 0) {
        printf("MPI_COMM_WORLD Size: %d\n", commSize);
        printf("MPI version: %d.%d\n", version, subversion);
        printf("MPI processor name: %s\n", processorName);
    }

    if (!UAT_MODE && argc != 3) {
        printf("Usage: heat <blocks_per_dimension> <time_steps>\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int topology_dimension[2];
    int period[2] = {FALSE, FALSE};
    int reorder = TRUE;
    MPI_Comm cartesian_comm;
    int cartId;

    MPI_Dims_create(commSize, DIMENSIONALITY, topology_dimension);
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, topology_dimension, period, reorder, &cartesian_comm);
    MPI_Comm_rank(cartesian_comm, &cartId);

    MPI_Cart_shift(cartesian_comm, 1, 1, &neighbors[EAST], &neighbors[WEST]);
    MPI_Cart_shift(cartesian_comm, 0, 1, &neighbors[NORTH], &neighbors[SOUTH]);

    int coords[2];
    MPI_Cart_coords(cartesian_comm, currentRank, DIMENSIONALITY, coords);
    printf("P:%d My coordinates are %d %d. My left is %d, my right is %d, my bottom is %d and my top is %d\n",
           currentRank, coords[0], coords[1], neighbors[EAST], neighbors[WEST], neighbors[SOUTH], neighbors[NORTH]);

    MPI_Datatype rowType;
    MPI_Datatype columnType;

    MPI_Type_vector(NY_HEAT + 2, 1, NX_HEAT + 2, MPI_FLOAT, &columnType);
    MPI_Type_vector(NX_HEAT + 2, 1, 1, MPI_FLOAT, &rowType);
    MPI_Type_commit(&columnType);
    MPI_Type_commit(&rowType);

    int currentStep;
    int currentNeighbor;
    int currentX, currentY;

    int localConvergence;
    int globalConvergence = FALSE;

    for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
        //North neighbor
        MPI_Send_init(&grid[currentGrid][1][0], 1, rowType, neighbors[NORTH], SOUTH, cartesian_comm, &request[SEND][currentGrid][NORTH]);
        MPI_Recv_init(&grid[currentGrid][0][0], 1, rowType, neighbors[NORTH], NORTH, cartesian_comm, &request[RECEIVE][currentGrid][NORTH]);
        //South neighbor
        MPI_Send_init(&grid[currentGrid][NY_HEAT][0], 1, rowType, neighbors[SOUTH], NORTH, cartesian_comm, &request[SEND][currentGrid][SOUTH]);
        MPI_Recv_init(&grid[currentGrid][NY_HEAT + 1][0], 1, rowType, neighbors[SOUTH], SOUTH, cartesian_comm, &request[RECEIVE][currentGrid][SOUTH]);
        //West neighbor
        MPI_Send_init(&grid[currentGrid][0][1], 1, columnType, neighbors[WEST], EAST, cartesian_comm, &request[SEND][currentGrid][WEST]);
        MPI_Recv_init(&grid[currentGrid][0][0], 1, columnType, neighbors[WEST], WEST, cartesian_comm, &request[RECEIVE][currentGrid][WEST]);
        //East neighbor
        MPI_Send_init(&grid[currentGrid][0][NX_HEAT], 1, columnType, neighbors[EAST], WEST, cartesian_comm, &request[SEND][currentGrid][EAST]);
        MPI_Recv_init(&grid[currentGrid][0][NX_HEAT + 1], 1, columnType, neighbors[EAST], EAST, cartesian_comm, &request[RECEIVE][currentGrid][EAST]);
    }

    startTime = MPI_Wtime();
    for (currentStep = 0; currentStep < EXEC_STEPS; ++currentStep) {

        for (currentNeighbor = 0; currentNeighbor < 4; ++currentNeighbor) {
            MPI_Start(&request[SEND][currentGrid][currentNeighbor]);
            MPI_Start(&request[RECEIVE][currentGrid][currentNeighbor]);
        }

        updateInner(2, NY_HEAT - 1, 2, &grid[currentGrid][0][0], &grid[1 - currentGrid][0][0]);

        MPI_Waitall(4, request[RECEIVE][currentGrid], MPI_STATUS_IGNORE);

        updateOuter(2, NY_HEAT - 1, 2, &grid[currentGrid][0][0], &grid[1 - currentGrid][0][0]);

        if (convergenceCheck && currentStep % CONVERGENCE_FREQ_STEPS == 0) {
            localConvergence = TRUE;
            for (currentX = 0; currentX < NX_HEAT; ++currentX)
                for (currentY = 0; currentY < NY_HEAT; ++currentY)
                    if (fabs(grid[1 - currentGrid][currentX][currentY] - grid[currentGrid][currentX][currentY]) >= 1e-3) {
                        localConvergence = FALSE;
                        break;
                    }

            MPI_Allreduce(&localConvergence, &globalConvergence, 1, MPI_INT, MPI_LAND, cartesian_comm);
        }

        MPI_Waitall(4, request[SEND][currentGrid], MPI_STATUS_IGNORE);

        if (globalConvergence == TRUE) break;
        currentGrid = 1 - currentGrid;
    }
    endTime = MPI_Wtime();

    printf("That took %f seconds\n", endTime - startTime);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

inline void updateInner(int start, int end, int ny, float *u1, float *u2) {
    int ix, iy;

#pragma omp parallel for shared(u1, u2) schedule(static) collapse(2)
    for (ix = start; ix <= end; ix++)
        for (iy = 1; iy <= ny - 2; iy++)
            *(u2 + ix * ny + iy) = *(u1 + ix * ny + iy) +
                                   parms.cx * (*(u1 + (ix + 1) * ny + iy) +
                                               *(u1 + (ix - 1) * ny + iy) -
                                               2.0 * *(u1 + ix * ny + iy)) +
                                   parms.cy * (*(u1 + ix * ny + iy + 1) +
                                               *(u1 + ix * ny + iy - 1) -
                                               2.0 * *(u1 + ix * ny + iy));
}

inline void updateOuter(int start, int end, int ny, float *u1, float *u2) {
    int ix, iy;

#pragma omp parallel for shared(u1, u2) schedule(static) collapse(2)
    for (ix = start; ix <= end; ix++)
        for (iy = 1; iy <= ny - 2; iy++)
            *(u2 + ix * ny + iy) = *(u1 + ix * ny + iy) +
                                   parms.cx * (*(u1 + (ix + 1) * ny + iy) +
                                               *(u1 + (ix - 1) * ny + iy) -
                                               2.0 * *(u1 + ix * ny + iy)) +
                                   parms.cy * (*(u1 + ix * ny + iy + 1) +
                                               *(u1 + ix * ny + iy - 1) -
                                               2.0 * *(u1 + ix * ny + iy));
}
