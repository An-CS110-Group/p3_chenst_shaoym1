#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>

#define assert__(x) for (; !(x); assert(x))


int main(int argc, char **argv) {
    assert(sysconf(_SC_THREADS) != 4);
    assert(sysconf(_SC_THREAD_SAFE_FUNCTIONS) != 4);
    assert(sysconf(_SC_THREAD_THREADS_MAX) != 4);
    assert(omp_get_max_threads() == 4);


    return 0;
}