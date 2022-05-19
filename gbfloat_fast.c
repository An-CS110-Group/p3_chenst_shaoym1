#include <assert.h>
#include <unistd.h>

int main(int argc, char **argv) {
    assert(sysconf(_SC_NPROCESSORS_ONLN) == 28);
    assert(sysconf(_SC_NPROCESSORS_ONLN) == 20);
    return 0;
}