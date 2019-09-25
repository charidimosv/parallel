cmake_minimum_required(VERSION 3.15)
project(parallel C)

add_subdirectory("${PROJECT_SOURCE_DIR}/src/mpi" "${PROJECT_SOURCE_DIR}/src/mpi/output")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/examples" "${PROJECT_SOURCE_DIR}/src/examples/output")
