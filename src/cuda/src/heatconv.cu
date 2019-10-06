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
} parms = {0.1, 0.1};

void printGridToFile(float *grid, const int totalRows, const int totalColumns, const char *fileName);

__global__ void iniData(const int totalRows, const int totalColumns, float *u1, float *u2);

__global__ void update(const int totalRows, const int totalColumns, const int currentConvergenceCheck, int *convergence, struct Parms *parms, float *oldGrid, float *nextGrid);

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

    int totalGridSize = totalRows * totalColumns;
    int totalGridBytesSize = sizeof(float) * totalGridSize;

    /////////////////////////////////////////////////////////////////////////////////
    //////////////                    CPU Variables                    //////////////
    /////////////////////////////////////////////////////////////////////////////////

    float *grid = (float *) malloc(totalGridBytesSize);
    int convergenceResult = 0;

    /////////////////////////////////////////////////////////////////////////////////
    //////////////                    GPU Variables                    //////////////
    /////////////////////////////////////////////////////////////////////////////////

    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 dimGrid(totalRows/THREADS_PER_BLOCK, totalColumns/THREADS_PER_BLOCK);

    float *oddGrid, *evenGrid;
    struct Parms *gpuParms = NULL;
    int *gpuConvergenceResult = NULL;

    cudaMalloc(&oddGrid, totalGridBytesSize);
    cudaMalloc(&evenGrid, totalGridBytesSize);

    iniData<<<dimGrid, dimBlock>>>(totalRows, totalColumns, oddGrid, evenGrid);

    cudaMalloc(&gpuParms, sizeof(struct Parms));
    cudaMalloc(&gpuConvergenceResult, sizeof(int));

    cudaMemcpy(gpuParms, &parms, sizeof(struct Parms), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuConvergenceResult, &convergenceResult, sizeof(int), cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////////////////////////////
    //////////////                    Main loop                    //////////////
    /////////////////////////////////////////////////////////////////////////////

    struct timeval timevalStart, timevalEnd;
    gettimeofday(&timevalStart, NULL);

    for (currentStep = 0; currentStep < steps; ++currentStep) {
        currentConvergenceCheck = convergenceCheck && currentStep % convFreqSteps == 0;

        if (currentStep % 2)
            update<<<dimGrid, dimBlock>>>(totalRows, totalColumns, currentConvergenceCheck, gpuConvergenceResult, gpuParms, evenGrid, oddGrid);
        else
            update<<<dimGrid, dimBlock>>>(totalRows, totalColumns, currentConvergenceCheck, gpuConvergenceResult, gpuParms, oddGrid, evenGrid);

        if (currentConvergenceCheck) {
            cudaMemcpy(&convergenceResult, gpuConvergenceResult, sizeof(int), cudaMemcpyDeviceToHost);
            if (convergenceResult) {
                convergenceStep = currentStep;
                break;
            }
        }
    }
    gettimeofday(&timevalEnd, NULL);

    printf("Results:\n");
    printf("- Runtime: %f sec\n", (timevalEnd.tv_sec - timevalStart.tv_sec) * 1000.0f + (timevalEnd.tv_usec - timevalStart.tv_usec) / 1000.0f);

    printf("- Convergence:\n");
    printf("-- checking: %s\n", convergenceCheck ? "YES" : "NO");
    printf("-- achieved: %s\n", convergenceResult ? "YES" : "NO");
    printf("-- at step: %d\n", convergenceStep);

    /////////////////////////////////////////////////////////////////////////////////
    //////////////                    Write to FIle                    //////////////
    /////////////////////////////////////////////////////////////////////////////////

    cudaMemcpy(grid, currentStep % 2 ? &oddGrid : &evenGrid, totalGridBytesSize, cudaMemcpyHostToDevice);
    printGridToFile(grid, totalRows, totalColumns, "final.dat");

    //////////////////////////////////////////////////////////////////////////////////
    //////////////                    Free Resources                    //////////////
    //////////////////////////////////////////////////////////////////////////////////

    free(grid);

    cudaFree(evenGrid);
    cudaFree(oddGrid);
    cudaFree(gpuParms);
    cudaFree(gpuConvergenceResult);

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

__global__ void iniData(const int totalRows, const int totalColumns, float *u1, float *u2) {

    int currentRow = (blockIdx.x * blockDim.x) + threadIdx.x;
    int currentColumn = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((currentRow >= 0 && currentRow < totalRows) && (currentColumn >= 0 && currentColumn < totalColumns)) {
        *(u1 + currentRow * totalColumns + currentColumn) = (float) (currentRow * (totalRows - currentRow - 1) * currentColumn * (totalColumns - currentColumn - 1));
        *(u2 + currentRow * totalColumns + currentColumn) = *(u1 + currentRow * totalColumns + currentColumn);
    }
}

__global__ void update(const int totalRows, const int totalColumns, const int currentConvergenceCheck, int *convergence, struct Parms *parms, float *oldGrid, float *nextGrid) {

    int currentRow = (blockIdx.x * blockDim.x) + threadIdx.x;
    int currentColumn = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((currentRow > 0 && currentRow < totalRows - 1) && (currentColumn > 0 && currentColumn < totalColumns - 1)) {
        *(nextGrid + currentRow * totalColumns + currentColumn) = *(oldGrid + currentRow * totalColumns + currentColumn) +
                                                                  parms->cx * (*(oldGrid + (currentRow + 1) * totalColumns + currentColumn) +
                                                                               *(oldGrid + (currentRow - 1) * totalColumns + currentColumn) -
                                                                               2.0 * *(oldGrid + currentRow * totalColumns + currentColumn)) +
                                                                  parms->cy * (*(oldGrid + currentRow * totalColumns + currentColumn + 1) +
                                                                               *(oldGrid + currentRow * totalColumns + currentColumn - 1) -
                                                                               2.0 * *(oldGrid + currentRow * totalColumns + currentColumn));

        if (currentConvergenceCheck && fabs(*(nextGrid + currentRow * totalColumns + currentColumn) - *(oldGrid + currentRow * totalColumns + currentColumn)) < 1e-2)
            *convergence = 1;
    }
}