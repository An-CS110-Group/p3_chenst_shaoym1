CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
project(p3_chenst_shaoym1)
set(CMAKE_C_FLAGS_DEBUG  "-g -O2 -fopenmp -pthread -mavx -mavx2")
set(CMAKE_C_FLAGS_RELEASE "-O2 -fopenmp -pthread -mavx -mavx2")

link_libraries(m)

add_executable(p3_chenst_shaoym1 gbfloat_fast.c stb_image.h stb_image_write.h)