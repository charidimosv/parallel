cmake_minimum_required(VERSION 3.14.5)
project(parallel C)

add_subdirectory("${PROJECT_SOURCE_DIR}/src/examples" "${PROJECT_SOURCE_DIR}/src/examples/output")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/mpi_original" "${PROJECT_SOURCE_DIR}/src/mpi_original/output")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/mpi" "${PROJECT_SOURCE_DIR}/src/mpi/output")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/cuda" "${PROJECT_SOURCE_DIR}/src/cuda/output")
