CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
project(Project3)
set(CMAKE_C_FLAGS "-O2 -fopenmp -pthread -march=native")

link_libraries(m)

add_executable(Project3 gbfloat_fast.c stb_image.h stb_image_write.h)