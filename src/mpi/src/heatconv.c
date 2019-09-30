#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define FALSE 0
#define TRUE 1

#define EXEC_STEPS 10000
#define CONVERGENCE_FREQ_STEPS 100

#define NORTH       0
#define SOUTH       1
#define WEST        2
#define EAST        3

#define DIMENSIONALITY  2
#define ROW             0
#define COLUMN          1
#define HALO_OFFSET     2

#define SEND        0
#define RECEIVE     1

#define UAT_MODE 1


struct Parms {
    float cx;
    float cy;
} parms = {0.1, 0.1};


int main(int argc, char **argv) {
    int commRank;
    int commSize;

    char processorName[MPI_MAX_PROCESSOR_NAME];
    int processorNameLen;

    int version, subversion;

    int convergenceCheck = 1;
    int fullProblemSize[DIMENSIONALITY];
    int subProblemSize[DIMENSIONALITY];

    int currentStep;
    int currentNeighbor;
    int currentConvergenceCheck;
    int currentRow, currentColumn;
    int currentGrid;

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    MPI Init/Print                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    MPI_Init(&argc, &argv);

    MPI_Get_version(&version, &subversion);
    MPI_Get_processor_name(processorName, &processorNameLen);

    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    if (commRank == 0) {
        printf("MPI_COMM_WORLD Size: %d\n", commSize);
        printf("MPI version: %d.%d\n", version, subversion);
        printf("MPI processor name: %s\n\n", processorName);
    }

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Argument Check                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    if (!UAT_MODE && argc != 3) {
        printf("Usage: heat <blocks_per_dimension> <time_steps>\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    } else {
        fullProblemSize[ROW] = atoi(argv[1]);
        fullProblemSize[COLUMN] = atoi(argv[2]);

        convergenceCheck = atoi(argv[3]);
    }

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Topology Setup                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    MPI_Comm cartComm;
    MPI_Request request[2][2][4];
    int neighbors[4];

    MPI_Datatype rowType;
    MPI_Datatype columnType;

    int topologyDimension[DIMENSIONALITY] = {0, 0};
    int period[DIMENSIONALITY] = {FALSE, FALSE};
    int reorder = TRUE;

    MPI_Dims_create(commSize, DIMENSIONALITY, topologyDimension);
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONALITY, topologyDimension, period, reorder, &cartComm);

    MPI_Cart_shift(cartComm, ROW, 1, &neighbors[NORTH], &neighbors[SOUTH]);
    MPI_Cart_shift(cartComm, COLUMN, 1, &neighbors[WEST], &neighbors[EAST]);

    int cartRank;
    MPI_Comm_rank(cartComm, &cartRank);

    int currentCoords[DIMENSIONALITY];
    MPI_Cart_coords(cartComm, commRank, DIMENSIONALITY, currentCoords);

    //    TODO - check correct splits
    subProblemSize[ROW] = fullProblemSize[ROW] / topologyDimension[ROW];
    subProblemSize[COLUMN] = fullProblemSize[COLUMN] / topologyDimension[COLUMN];

    int totalRows = subProblemSize[ROW] + HALO_OFFSET;
    int totalColumns = subProblemSize[COLUMN] + HALO_OFFSET;

    MPI_Type_vector(subProblemSize[COLUMN], 1, 1, MPI_FLOAT, &rowType);
    MPI_Type_vector(subProblemSize[ROW], 1, totalColumns, MPI_FLOAT, &columnType);
    MPI_Type_commit(&rowType);
    MPI_Type_commit(&columnType);

    ////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Initialise Main Grid                    //////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    float *u1, *u2;
    float **grid[2];
    for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
        grid[currentGrid] = (float **) malloc(sizeof(float *) * (totalRows));
        grid[currentGrid][0] = (float *) malloc(sizeof(float) * (totalRows * totalColumns));
        for (currentRow = 1; currentRow < totalRows; ++currentRow)
            grid[currentGrid][currentRow] = &grid[currentGrid][0][currentRow * totalColumns];
    }

    for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
        if (neighbors[NORTH] == MPI_PROC_NULL)
            for (currentColumn = 0; currentColumn < totalColumns; ++currentColumn)
                grid[currentGrid][0][currentColumn] = 0;
        if (neighbors[SOUTH] == MPI_PROC_NULL)
            for (currentColumn = 0; currentColumn < totalColumns; ++currentColumn)
                grid[currentGrid][totalRows - 1][currentColumn] = 0;

        if (neighbors[WEST] == MPI_PROC_NULL)
            for (currentRow = 0; currentRow < totalRows; ++currentRow)
                grid[currentGrid][currentRow][0] = 0;
        if (neighbors[EAST] == MPI_PROC_NULL)
            for (currentRow = 0; currentRow < totalRows; ++currentRow)
                grid[currentGrid][currentRow][totalColumns - 1] = 0;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Initialise Boarders                    //////////////
    ///////////////////////////////////////////////////////////////////////////////////////

    int splitterCount = 2 * subProblemSize[ROW] + 2 * subProblemSize[COLUMN] - 4;
    int *splitter[2];
    splitter[ROW] = (int *) malloc(sizeof(int) * splitterCount);
    splitter[COLUMN] = (int *) malloc(sizeof(int) * splitterCount);

    int tempCounter = 0;
    for (currentColumn = 1; currentColumn < subProblemSize[COLUMN] + 1; ++currentColumn) {
        splitter[ROW][tempCounter] = 1;
        splitter[COLUMN][tempCounter++] = currentColumn;
    }
    for (currentColumn = 1; currentColumn < subProblemSize[COLUMN] + 1; ++currentColumn) {
        splitter[ROW][tempCounter] = subProblemSize[ROW];
        splitter[COLUMN][tempCounter++] = currentColumn;
    }
    for (currentRow = 2; currentRow < subProblemSize[ROW]; ++currentRow) {
        splitter[ROW][tempCounter] = currentRow;
        splitter[COLUMN][tempCounter++] = 1;
    }
    for (currentRow = 2; currentRow < subProblemSize[ROW]; ++currentRow) {
        splitter[ROW][tempCounter] = currentRow;
        splitter[COLUMN][tempCounter++] = subProblemSize[COLUMN];
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Prepare send/receive requests                    //////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
        MPI_Send_init(&grid[currentGrid][1][1], 1, rowType, neighbors[NORTH], cartRank, cartComm, &request[SEND][currentGrid][NORTH]);
        MPI_Recv_init(&grid[currentGrid][0][1], 1, rowType, neighbors[NORTH], neighbors[NORTH], cartComm, &request[RECEIVE][currentGrid][NORTH]);

        MPI_Send_init(&grid[currentGrid][totalRows - 2][1], 1, rowType, neighbors[SOUTH], cartRank, cartComm, &request[SEND][currentGrid][SOUTH]);
        MPI_Recv_init(&grid[currentGrid][totalRows - 1][1], 1, rowType, neighbors[SOUTH], neighbors[SOUTH], cartComm, &request[RECEIVE][currentGrid][SOUTH]);

        MPI_Send_init(&grid[currentGrid][1][1], 1, columnType, neighbors[WEST], cartRank, cartComm, &request[SEND][currentGrid][WEST]);
        MPI_Recv_init(&grid[currentGrid][1][0], 1, columnType, neighbors[WEST], neighbors[WEST], cartComm, &request[RECEIVE][currentGrid][WEST]);

        MPI_Send_init(&grid[currentGrid][1][totalRows - 2], 1, columnType, neighbors[EAST], cartRank, cartComm, &request[SEND][currentGrid][EAST]);
        MPI_Recv_init(&grid[currentGrid][1][totalRows - 1], 1, columnType, neighbors[EAST], neighbors[EAST], cartComm, &request[RECEIVE][currentGrid][EAST]);
    }

    ///////////////////////////////////////////////////////////////////////////////
    //////////////                    Print Setup                    //////////////
    ///////////////////////////////////////////////////////////////////////////////

    printf("Printing boarders:\n");
    for (tempCounter = 0; tempCounter < splitterCount; ++tempCounter) {
        printf("\t%dx%d\n", splitter[ROW][tempCounter], splitter[COLUMN][tempCounter]);
    }

    printf("CommRank: %d, CartRank: %d, Coords: %dx%d. EAST: %d, WEST: %d, SOUTH: %d, NORTH: %d. My problem is %dx%d\n",
           commRank, cartRank, currentCoords[ROW], currentCoords[COLUMN], neighbors[EAST], neighbors[WEST], neighbors[SOUTH], neighbors[NORTH], subProblemSize[ROW],
           subProblemSize[COLUMN]);

    /////////////////////////////////////////////////////////////////////////////
    //////////////                    Main loop                    //////////////
    /////////////////////////////////////////////////////////////////////////////

    double startTime, endTime;

    currentGrid = 0;
    int localConvergence = TRUE;
    int globalConvergence = FALSE;

    MPI_Barrier(cartComm);
    startTime = MPI_Wtime();

//#pragma omp parallel default(none) private(currentStep, currentRow, currentColumn, tempCounter, u1, u2) shared(grid, splitter[ROW], splitter[COLUMN], currentConvergenceCheck, convergenceCheck, currentNeighbor, currentGrid, request, cartComm, globalConvergence, localConvergence, totalRows, totalColumns, splitterCount, parms, ompi_mpi_op_land, ompi_mpi_int)
#pragma omp parallel private(currentStep, currentRow, currentColumn, tempCounter, u1, u2)
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

            u1 = &grid[currentGrid][0][0];
            u2 = &grid[1 - currentGrid][0][0];

            if (currentConvergenceCheck) {
#pragma omp for schedule(static) collapse(DIMENSIONALITY) reduction(&&:localConvergence)
                for (currentRow = 2; currentRow < totalRows - 1; ++currentRow)
                    for (currentColumn = 2; currentColumn < totalColumns - 1; ++currentColumn) {
                        *(u2 + currentRow * totalColumns + currentColumn) = *(u1 + currentRow * totalColumns + currentColumn) +
                                                                            parms.cx * (*(u1 + (currentRow + 1) * totalColumns + currentColumn) +
                                                                                        *(u1 + (currentRow - 1) * totalColumns + currentColumn) -
                                                                                        2.0 * *(u1 + currentRow * totalColumns + currentColumn)) +
                                                                            parms.cy * (*(u1 + currentRow * totalColumns + currentColumn + 1) +
                                                                                        *(u1 + currentRow * totalColumns + currentColumn - 1) -
                                                                                        2.0 * *(u1 + currentRow * totalColumns + currentColumn));
                        if (fabs(*(u2 + currentRow * totalColumns + currentColumn) - *(u1 + currentRow * totalColumns + currentColumn)) >= 1e-3) {
                            localConvergence = FALSE;
                        }
                    }
            } else {
#pragma omp for schedule(static) collapse(DIMENSIONALITY)
                for (currentRow = 2; currentRow < totalRows - 1; ++currentRow)
                    for (currentColumn = 2; currentColumn < totalColumns - 1; ++currentColumn) {
                        *(u2 + currentRow * totalColumns + currentColumn) = *(u1 + currentRow * totalColumns + currentColumn) +
                                                                            parms.cx * (*(u1 + (currentRow + 1) * totalColumns + currentColumn) +
                                                                                        *(u1 + (currentRow - 1) * totalColumns + currentColumn) -
                                                                                        2.0 * *(u1 + currentRow * totalColumns + currentColumn)) +
                                                                            parms.cy * (*(u1 + currentRow * totalColumns + currentColumn + 1) +
                                                                                        *(u1 + currentRow * totalColumns + currentColumn - 1) -
                                                                                        2.0 * *(u1 + currentRow * totalColumns + currentColumn));
                    }
            }

#pragma omp single
            {
                MPI_Waitall(4, request[RECEIVE][currentGrid], MPI_STATUS_IGNORE);
            }

            if (currentConvergenceCheck) {
#pragma omp for schedule(static) reduction(&&:localConvergence)
                for (tempCounter = 0; tempCounter < splitterCount; ++tempCounter) {
                    currentRow = splitter[ROW][tempCounter];
                    currentColumn = splitter[COLUMN][tempCounter];
                    *(u2 + currentRow * totalColumns + currentColumn) = *(u1 + currentRow * totalColumns + currentColumn) +
                                                                        parms.cx * (*(u1 + (currentRow + 1) * totalColumns + currentColumn) +
                                                                                    *(u1 + (currentRow - 1) * totalColumns + currentColumn) -
                                                                                    2.0 * *(u1 + currentRow * totalColumns + currentColumn)) +
                                                                        parms.cy * (*(u1 + currentRow * totalColumns + currentColumn + 1) +
                                                                                    *(u1 + currentRow * totalColumns + currentColumn - 1) -
                                                                                    2.0 * *(u1 + currentRow * totalColumns + currentColumn));
                    if (fabs(*(u2 + currentRow * totalColumns + currentColumn) - *(u1 + currentRow * totalColumns + currentColumn)) >= 1e-3) {
                        localConvergence = FALSE;
                    }
                }
            } else {
#pragma omp for schedule(static)
                for (tempCounter = 0; tempCounter < splitterCount; ++tempCounter) {
                    currentRow = splitter[ROW][tempCounter];
                    currentColumn = splitter[COLUMN][tempCounter];
                    *(u2 + currentRow * totalColumns + currentColumn) = *(u1 + currentRow * totalColumns + currentColumn) +
                                                                        parms.cx * (*(u1 + (currentRow + 1) * totalColumns + currentColumn) +
                                                                                    *(u1 + (currentRow - 1) * totalColumns + currentColumn) -
                                                                                    2.0 * *(u1 + currentRow * totalColumns + currentColumn)) +
                                                                        parms.cy * (*(u1 + currentRow * totalColumns + currentColumn + 1) +
                                                                                    *(u1 + currentRow * totalColumns + currentColumn - 1) -
                                                                                    2.0 * *(u1 + currentRow * totalColumns + currentColumn));

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

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Free Resources                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    free(splitter[ROW]);
    free(splitter[COLUMN]);

    for (currentGrid = 0; currentGrid < 2; ++currentGrid) {
        for (currentRow = 0; currentRow < totalRows; ++currentRow)
            free(grid[currentGrid][currentRow]);
        free(grid[currentGrid]);
    }

    MPI_Type_free(&rowType);
    MPI_Type_free(&columnType);
    MPI_Comm_free(&cartComm);

    ////////////////////////////////////////////////////////////////////////////
    //////////////                    Finalize                    //////////////
    ////////////////////////////////////////////////////////////////////////////

    MPI_Finalize();

    return EXIT_SUCCESS;
}