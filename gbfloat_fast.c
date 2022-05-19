#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include <sys/sysinfo.h>

#define assert__(x) for (; !(x); assert(x))


int main(int argc, char **argv) {
    printf("%d %d", get_nprocs_conf(), get_nprocs());
    assert(get_nprocs_conf() == 14);
    assert(get_nprocs() == 14);


    return 0;
}