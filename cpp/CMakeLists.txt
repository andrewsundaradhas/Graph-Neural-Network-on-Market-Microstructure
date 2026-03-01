cmake_minimum_required(VERSION 3.10)
project(gnn_lob_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
# Set CMAKE_PREFIX_PATH to the folder containing LibTorch
# example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -march=native")

include_directories(${TORCH_INCLUDE_DIRS})

add_executable(gnn_lob_inference 
    main.cpp 
    graph_builder.cpp 
    model_wrapper.cpp 
    strategy.cpp
)

target_link_libraries(gnn_lob_inference "${TORCH_LIBRARIES}")

if (MSVC)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib/ DESTINATION ${CMAKE_BINARY_DIR} FILES_MATCHING PATTERN "*.dll")
endif()
