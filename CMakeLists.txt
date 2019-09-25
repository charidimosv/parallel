cmake_minimum_required(VERSION 3.7)
project(parallel)

add_subdirectory("${PROJECT_SOURCE_DIR}/src/mpi" "${PROJECT_SOURCE_DIR}/src/mpi/output")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/openmp" "${PROJECT_SOURCE_DIR}/src/openmp/output")
