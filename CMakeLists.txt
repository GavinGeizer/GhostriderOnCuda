cmake_minimum_required(VERSION 3.24)
project(GhostriderOnCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(ghostrider
    cuda/algorithms.cu
    src/pipeline.cu
)

target_include_directories(ghostrider PUBLIC include)

add_executable(ghostrider_prototype src/main.cu)
target_link_libraries(ghostrider_prototype PRIVATE ghostrider)
