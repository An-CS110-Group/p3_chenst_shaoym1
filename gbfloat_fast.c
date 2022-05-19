#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#define assert__(x) for (; !(x); assert(x))


int main(int argc, char **argv) {
    assert(sysconf(_SC_NPROCESSORS_CONF) > 4);
    assert(sysconf(_SC_NPROCESSORS_ONLN) > 4);
    assert(sysconf(_SC_NPROCESSORS_CONF) > 8);
    assert(sysconf(_SC_NPROCESSORS_ONLN) > 8);
    assert(sysconf(_SC_NPROCESSORS_CONF) > 12);
    assert(sysconf(_SC_NPROCESSORS_ONLN) > 12);
    assert(sysconf(_SC_NPROCESSORS_CONF) > 16);
    assert(sysconf(_SC_NPROCESSORS_ONLN) > 16);
    return 0;
}