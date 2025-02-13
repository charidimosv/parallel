set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp")

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} -lm)

if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
endif ()

if (MPI_COMPILE_FLAGS)
    set(CMAKE_C_COMPILE_FLAGS ${CMAKE_C_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
endif ()

if (MPI_LINK_FLAGS)
    set(CMAKE_C_LINK_FLAGS ${CMAKE_C_LINK_FLAGS} ${MPI_LINK_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
endif ()
