cmake_minimum_required(VERSION 3.22)

project(numC)

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCES main.cpp)

include_directories(openblas/include)
link_directories(openblas/lib)

add_executable(main ${SOURCES})

# Set the output directory for executables
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main libopenblas.lib)