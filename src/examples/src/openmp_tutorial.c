#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int sum = 0;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int i;

#pragma omp single
        printf("omp before - thread = %d\n", tid);

#pragma omp for
        for (i = 0; i < 3; ++i)
            printf("omp for - thread = %d, i = %d\n", tid, i);
#pragma omp master
        printf("omp after - thread = %d\n", tid);

//#pragma omp for reduction(+:sum)
//        for (i = 0; i < 4; i++) {
//            printf("omp for - thread = %d, i = %d\n", tid, i);
//            sum += i;
//        }
//
//#pragma omp single
//        {
//            printf("omp single - thread = %d, sum = %d\n\n", tid, sum);
//        }
//
//#pragma omp master
//        {
//            printf("omp master - thread = %d, sum = %d\n\n", tid, sum);
//        }

    }
}
