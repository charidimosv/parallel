#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#ifdef _OPENMP

#include <omp.h>

#endif

#define FALSE 0
#define TRUE 1

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

    int commRank;
    int commSize;

    char processorName[MPI_MAX_PROCESSOR_NAME];
    int processorNameLen;

    int version, subversion;

    float grid[2][NX_HEAT + 2][NY_HEAT + 2];

    int convergenceCheck = 1;

    ////////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    MPI Init/Print                    ////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Init(&argc, &argv);

    MPI_Get_version(&version, &subversion);
    MPI_Get_processor_name(processorName, &processorNameLen);

    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    if (commRank == 0) {
        printf("MPI_COMM_WORLD Size: %d\n", commSize);
        printf("MPI version: %d.%d\n", version, subversion);
        printf("MPI processor name: %s\n", processorName);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Argument Check                    ////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////

    if (!UAT_MODE && argc != 3) {
        printf("Usage: heat <blocks_per_dimension> <time_steps>\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Topology Setup                    ////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Comm cartComm;
    MPI_Request request[2][2][4];

    MPI_Datatype rowType;
    MPI_Datatype columnType;

    int currentCoords[2];
    int neighbors[4];

    int topologyDimension[2];
    int period[2] = {FALSE, FALSE};
    int reorder = TRUE;
    int cartRank;

    MPI_Dims_create(commSize, DIMENSIONALITY, topologyDimension);
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, topologyDimension, period, reorder, &cartComm);
    MPI_Comm_rank(cartComm, &cartRank);

    MPI_Cart_coords(cartComm, commRank, DIMENSIONALITY, currentCoords);
    MPI_Cart_shift(cartComm, 1, 1, &neighbors[EAST], &neighbors[WEST]);
    MPI_Cart_shift(cartComm, 0, 1, &neighbors[NORTH], &neighbors[SOUTH]);

    printf("CommRank: %d, CartRank: %d, Coords: %dx%d. EAST: %d, WEST: %d, SOUTH: %d, NORTH: %d\n",
           commRank, cartRank, currentCoords[0], currentCoords[1], neighbors[EAST], neighbors[WEST], neighbors[SOUTH], neighbors[NORTH]);

    MPI_Type_vector(NY_HEAT + 2, 1, NX_HEAT + 2, MPI_FLOAT, &columnType);
    MPI_Type_vector(NX_HEAT + 2, 1, 1, MPI_FLOAT, &rowType);
    MPI_Type_commit(&columnType);
    MPI_Type_commit(&rowType);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Prepare send/receive requests                    ////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    int currentGrid = 0;
    for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
        MPI_Send_init(&grid[currentGrid][1][0], 1, rowType, neighbors[NORTH], SOUTH, cartComm, &request[SEND][currentGrid][NORTH]);
        MPI_Recv_init(&grid[currentGrid][0][0], 1, rowType, neighbors[NORTH], NORTH, cartComm, &request[RECEIVE][currentGrid][NORTH]);

        MPI_Send_init(&grid[currentGrid][NY_HEAT][0], 1, rowType, neighbors[SOUTH], NORTH, cartComm, &request[SEND][currentGrid][SOUTH]);
        MPI_Recv_init(&grid[currentGrid][NY_HEAT + 1][0], 1, rowType, neighbors[SOUTH], SOUTH, cartComm, &request[RECEIVE][currentGrid][SOUTH]);

        MPI_Send_init(&grid[currentGrid][0][1], 1, columnType, neighbors[WEST], EAST, cartComm, &request[SEND][currentGrid][WEST]);
        MPI_Recv_init(&grid[currentGrid][0][0], 1, columnType, neighbors[WEST], WEST, cartComm, &request[RECEIVE][currentGrid][WEST]);

        MPI_Send_init(&grid[currentGrid][0][NX_HEAT], 1, columnType, neighbors[EAST], WEST, cartComm, &request[SEND][currentGrid][EAST]);
        MPI_Recv_init(&grid[currentGrid][0][NX_HEAT + 1], 1, columnType, neighbors[EAST], EAST, cartComm, &request[RECEIVE][currentGrid][EAST]);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Main loop                    ////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

    int currentStep;
    int currentNeighbor;
    int currentConvergenceCheck;
    int currentX, currentY;
    currentGrid = 0;

    double startTime, endTime;

    int localConvergence = TRUE;
    int globalConvergence = FALSE;

    MPI_Barrier(cartComm);
    startTime = MPI_Wtime();

#pragma omp parallel
    {
        for (currentStep = 0; currentStep < EXEC_STEPS; ++currentStep) {

#pragma omp single
            {
                currentConvergenceCheck = convergenceCheck && currentStep % CONVERGENCE_FREQ_STEPS == 0;

                for (currentNeighbor = 0; currentNeighbor < 4; ++currentNeighbor) {
                    MPI_Start(&request[SEND][currentGrid][currentNeighbor]);
                    MPI_Start(&request[RECEIVE][currentGrid][currentNeighbor]);
                }
            }

            updateInner(2, NY_HEAT - 1, 2, &grid[currentGrid][0][0], &grid[1 - currentGrid][0][0]);

#pragma omp single
            {
                MPI_Waitall(4, request[RECEIVE][currentGrid], MPI_STATUS_IGNORE);
            }

            updateOuter(2, NY_HEAT - 1, 2, &grid[currentGrid][0][0], &grid[1 - currentGrid][0][0]);

            if (currentConvergenceCheck) {
#pragma omp for reduction(&&:localConvergence)
                {
                    for (currentX = 0; currentX < NX_HEAT; ++currentX)
                        for (currentY = 0; currentY < NY_HEAT; ++currentY)
                            if (fabs(grid[1 - currentGrid][currentX][currentY] - grid[currentGrid][currentX][currentY]) >= 1e-3) {
                                localConvergence && = FALSE;
                                break;
                            }
                }
            }

#pragma omp single
            {
                if (currentConvergenceCheck) {
                    MPI_Allreduce(&localConvergence, &globalConvergence, 1, MPI_INT, MPI_LAND, cartComm);
                    localConvergence = TRUE;
                }

                MPI_Waitall(4, request[SEND][currentGrid], MPI_STATUS_IGNORE);
                currentGrid = 1 - currentGrid;
            }

            if (globalConvergence == TRUE) break;
        }
    }

    endTime = MPI_Wtime();
    printf("That took %f seconds\n", endTime - startTime);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Finalize                    ////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

    MPI_Finalize();
    return EXIT_SUCCESS;
}

inline void updateInner(int start, int end, int ny, float *u1, float *u2) {
    int ix, iy;

#pragma omp for schedule(static) collapse(2)
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

#pragma omp for schedule(static) collapse(2)
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
