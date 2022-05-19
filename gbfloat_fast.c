#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#define assert__(x) for (; !(x); assert(x))


int main(int argc, char **argv) {
    printf("%d %d", get_nprocs_conf(), get_nprocs());
    assert(get_nprocs_conf() >=1);
    assert(get_nprocs_conf() >=2);
    assert(get_nprocs_conf() >=3);
    assert(get_nprocs_conf() >=4);
    assert(get_nprocs_conf() >=5);
    assert(get_nprocs_conf() >=6);
    assert(get_nprocs_conf() >=7);
    assert(get_nprocs_conf() >=8);
    assert(get_nprocs_conf() >=9);
    assert(get_nprocs_conf() >=10);
    assert(get_nprocs_conf() >=11);
    assert(get_nprocs_conf() >=12);
    assert(get_nprocs_conf() >=13);
    assert(get_nprocs_conf() >=14);
    assert(get_nprocs_conf() >=15);
    assert(get_nprocs_conf() >=16);
    assert(get_nprocs_conf() >=17);
    assert(get_nprocs_conf() >=18);


    assert(get_nprocs() == 14);


    return 0;
}