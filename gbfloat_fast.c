#include <assert.h>
#include <unistd.h>

int main(int argc, char **argv) {
    assert(sysconf(_SC_NPROCESSORS_ONLN) >= 4);
    assert(sysconf(_SC_NPROCESSORS_ONLN) == 4);
    return 0;
}