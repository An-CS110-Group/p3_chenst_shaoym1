#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#define assert__(x) for (; !(x); assert(x))


int main(int argc, char **argv) {
    printf("%d %d", get_nprocs_conf(), get_nprocs());
    assert(get_nprocs() >= 1);
    assert(get_nprocs() >= 2);
    assert(get_nprocs() >= 3);
    assert(get_nprocs() >= 4);
    assert(get_nprocs() >= 5);
    assert(get_nprocs() >= 6);
    assert(get_nprocs() >= 7);
    assert(get_nprocs() >= 8);
    assert(get_nprocs() >= 9);
    assert(get_nprocs() >= 10);
    assert(get_nprocs() >= 11);
    assert(get_nprocs() >= 12);
    assert(get_nprocs() >= 13);
    assert(get_nprocs() >= 14);
    assert(get_nprocs() >= 15);
    assert(get_nprocs() >= 16);
    assert(get_nprocs() >= 17);
    assert(get_nprocs() >= 18);
    assert(get_nprocs() <= 50);


    assert(get_nprocs() == 14);


    return 0;
}