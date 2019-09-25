#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int myid, numprocs, message;
    MPI_Status recv_status, send_status;
    int next, last, in;
    MPI_Request request;
    int tag = 23; /* arbitrary value */

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    /* work out identity of neighbours */
    next = (myid + 1) % numprocs;
    last = (myid + numprocs - 1) % numprocs;
    in = -1;
    message = myid;
    /* messages circle until each process receives its own id back again */
    while (in != myid) {
        MPI_Isend(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD, &request);
        MPI_Recv(&in, 1, MPI_INT, last, tag, MPI_COMM_WORLD, &recv_status);
        printf("Process %d received %d\n", myid, in);
        MPI_Wait(&request, &send_status);
        message = in;
    };
    MPI_Finalize();
}