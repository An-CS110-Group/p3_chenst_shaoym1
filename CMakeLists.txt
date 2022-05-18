CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
project(p3_chenst_shaoym1)
set(CMAKE_C_FLAGS_DEBUG  "-g -O0 -fopenmp -pthread -mavx -mavx2 -mfma")
set(CMAKE_C_FLAGS_RELEASE "-O2 -fopenmp -pthread -mavx -mavx2 -mfma")

link_libraries(m)

add_executable(p3_chenst_shaoym1 gbfloat_fast.c stb_image.h stb_image_write.h)
add_executable(test_accuracy test_accuracy.c stb_image.h stb_image_write.h)