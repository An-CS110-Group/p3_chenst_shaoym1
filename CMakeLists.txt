CMAKE_MINIMUM_REQUIRED(3.20)
project(Project3)
set(CMAKE_CXX_FLAGS "-O2 -fopenmp -pthread")

link_libraries(m)
add_executable(Project3 )