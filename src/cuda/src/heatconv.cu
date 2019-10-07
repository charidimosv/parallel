#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define FALSE                   0
#define TRUE                    1

#define THREADS_PER_BLOCK       32

struct Parms {
    float cx;
    float cy;
};

void printGridToFile(float *grid, const int totalRows, const int totalColumns, const char *fileName);

__global__
void iniData(const int totalRows, const int totalColumns, float *u1, float *u2);

__global__
void update(const int totalRows, const int totalColumns, const int currentConvergenceCheck, int *convergence, struct Parms *parms, float *oldGrid, float *nextGrid);

int main(int argc, char **argv) {

    int steps;
    int convFreqSteps;
    int totalRows, totalColumns;
    int convergenceCheck;

    int currentStep;
    int currentConvergenceCheck;
    int convergenceStep = -1;

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Argument Check                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    if (argc == 5) {
        steps = atoi(argv[1]);
        totalRows = atoi(argv[2]);
        totalColumns = atoi(argv[3]);
        convergenceCheck = atoi(argv[4]);
    } else {
        printf("Usage: heatconv <ROWS> <COLUMNS> <CONVERGENCE_FLAG>\n");
        exit(EXIT_FAILURE);
    }
    convFreqSteps = (int) sqrt(steps);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////                    Unified Memory – accessible from CPU or GPU                    //////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int totalGridSize = totalRows * totalColumns;
    unsigned int totalGridBytesSize = sizeof(float) * totalGridSize;

    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 dimGrid((totalRows + dimBlock.x - 1) / dimBlock.x, (totalColumns + dimBlock.y - 1) / dimBlock.y);

    float *gridOdd, *gridEven;
    cudaMallocManaged(&gridOdd, totalGridBytesSize);
    cudaMallocManaged(&gridEven, totalGridBytesSize);
    iniData<<<dimGrid, dimBlock>>>(totalRows, totalColumns, gridOdd, gridEven);

    cudaDeviceSynchronize();

    struct Parms *parms;
    cudaMallocManaged(&parms, sizeof(struct Parms));
    parms->cx = 0.1f;
    parms->cy = 0.1f;

    int *convergenceResult;
    cudaMallocManaged(&convergenceResult, sizeof(int));
    *convergenceResult = 1;

    /////////////////////////////////////////////////////////////////////////////
    //////////////                    Main loop                    //////////////
    /////////////////////////////////////////////////////////////////////////////

    struct timeval timevalStart, timevalEnd;
    gettimeofday(&timevalStart, NULL);

    for (currentStep = 0; currentStep < steps; ++currentStep) {
        currentConvergenceCheck = convergenceCheck && currentStep % convFreqSteps == 0;

        if (currentStep % 2)
            update<<<dimGrid, dimBlock>>>(totalRows, totalColumns, currentConvergenceCheck, convergenceResult, parms, gridEven, gridOdd);
        else
            update<<<dimGrid, dimBlock>>>(totalRows, totalColumns, currentConvergenceCheck, convergenceResult, parms, gridOdd, gridEven);

        cudaDeviceSynchronize();

        if (currentConvergenceCheck) {
            if (*convergenceResult) {
                convergenceStep = currentStep;
                break;
            } else *convergenceResult = 1;
        }
    }

    gettimeofday(&timevalEnd, NULL);

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Gather Results                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    printf("Results:\n");
    printf("- Runtime: %f sec\n", (float) (timevalEnd.tv_sec - timevalStart.tv_sec) * 1000.0f + (float) (timevalEnd.tv_usec - timevalStart.tv_usec) / 1000.0f);

    printf("- Convergence:\n");
    printf("-- checking: %s\n", convergenceCheck ? "YES" : "NO");
    printf("-- achieved: %s\n", *convergenceResult ? "YES" : "NO");
    printf("-- at step: %d\n", convergenceStep);

    /////////////////////////////////////////////////////////////////////////////////
    //////////////                    Write to FIle                    //////////////
    /////////////////////////////////////////////////////////////////////////////////

    printGridToFile(currentStep % 2 ? gridOdd : gridEven, totalRows, totalColumns, "final.dat");

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Free Resources                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    cudaFree(gridEven);
    cudaFree(gridOdd);
    cudaFree(parms);
    cudaFree(convergenceResult);

    ////////////////////////////////////////////////////////////////////////////
    //////////////                    Finalize                    //////////////
    ////////////////////////////////////////////////////////////////////////////

    return EXIT_SUCCESS;
}

void printTable(float **grid, int totalRows, int totalColumns) {
    printf("\n");
    for (int currentRow = 0; currentRow < totalRows; ++currentRow) {
        for (int currentColumn = 0; currentColumn < totalColumns; ++currentColumn) {
            printf("%.1f\t", grid[currentRow][currentColumn]);
        }
        printf("\n");
    }
    printf("\n");
}

void printGridToFile(float *grid, const int totalRows, const int totalColumns, const char *fileName) {
    int currentRow, currentColumn;
    FILE *fp;
    printf("Writing to file %s...\n", fileName);

    fp = fopen(fileName, "w");
    for (currentRow = 1; currentRow < totalRows - 1; ++currentRow) {
        for (currentColumn = 1; currentColumn < totalColumns - 1; ++currentColumn)
            fprintf(fp, "\t%6.1f\t", grid[currentRow * totalColumns + currentColumn]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

__global__
void iniData(const int totalRows, const int totalColumns, float *u1, float *u2) {
    int currentRow = blockIdx.x * blockDim.x + threadIdx.x;
    int currentColumn = blockIdx.y * blockDim.y + threadIdx.y;

    if ((currentRow >= 0 && currentRow < totalRows) && (currentColumn >= 0 && currentColumn < totalColumns)) {
        *(u1 + currentRow * totalColumns + currentColumn) = (float) (currentRow * (totalRows - currentRow - 1) * currentColumn * (totalColumns - currentColumn - 1));
        *(u2 + currentRow * totalColumns + currentColumn) = *(u1 + currentRow * totalColumns + currentColumn);
    }
}

__global__
void update(const int totalRows, const int totalColumns, const int currentConvergenceCheck, int *convergence, struct Parms *parms, float *oldGrid, float *nextGrid) {
    int currentRow = blockIdx.x * blockDim.x + threadIdx.x;
    int currentColumn = blockIdx.y * blockDim.y + threadIdx.y;

    if (currentRow > 0 && currentRow < totalRows - 1 && currentColumn > 0 && currentColumn < totalColumns - 1) {
        *(nextGrid + currentRow * totalColumns + currentColumn) = *(oldGrid + currentRow * totalColumns + currentColumn) +
                                                                  parms->cx * (*(oldGrid + (currentRow + 1) * totalColumns + currentColumn) +
                                                                               *(oldGrid + (currentRow - 1) * totalColumns + currentColumn) -
                                                                               2.0 * *(oldGrid + currentRow * totalColumns + currentColumn)) +
                                                                  parms->cy * (*(oldGrid + currentRow * totalColumns + currentColumn + 1) +
                                                                               *(oldGrid + currentRow * totalColumns + currentColumn - 1) -
                                                                               2.0 * *(oldGrid + currentRow * totalColumns + currentColumn));

        if (currentConvergenceCheck && fabs((double) *(nextGrid + currentRow * totalColumns + currentColumn) - *(oldGrid + currentRow * totalColumns + currentColumn)) > 1e-2)
            *convergence = 0;
    }
}