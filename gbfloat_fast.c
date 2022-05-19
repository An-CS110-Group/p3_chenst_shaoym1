#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#define assert__(x) for (; !(x); assert(x))


int main(int argc, char **argv) {
    assert(sysconf(_SC_THREAD_THREADS_MAX) > 4);
    //    assert(sysconf(_SC_NPROCESSORS_ONLN) > 4);
    assert(sysconf(_SC_THREAD_THREADS_MAX) > 8);
    //    assert(sysconf(_SC_NPROCESSORS_ONLN) > 8);
    assert(sysconf(_SC_THREAD_THREADS_MAX) > 12);
    //    assert(sysconf(_SC_NPROCESSORS_ONLN) > 12);
    assert(sysconf(_SC_THREAD_THREADS_MAX) > 16);
    //    assert(sysconf(_SC_NPROCESSORS_ONLN) > 16);
    assert(sysconf(_SC_THREAD_THREADS_MAX) > 20);
    return 0;
}