#include <assert.h>
#include <unistd.h>

int main(int argc, char **argv) {
    assert(sysconf(_SC_NPROCESSORS_CONF) == 14);
    assert(sysconf(_SC_NPROCESSORS_CONF) == 10);
    return 0;
}