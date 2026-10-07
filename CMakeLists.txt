cmake_minimum_required(VERSION 3.24)
project(gpu_vision_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GVE_ENABLE_CUDA "Enable CUDA automatically when a compiler is available" ON)
option(GVE_BUILD_TESTS "Build tests" ON)
option(GVE_BUILD_BENCHMARKS "Build benchmarks" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

include(CheckLanguage)
set(GVE_HAS_CUDA OFF)
if(GVE_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GVE_HAS_CUDA ON)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
  else()
    message(STATUS "CUDA compiler not found; building CPU/OpenCV mode")
  endif()
endif()

add_library(gve_cpu
  src/cpu/gaussian_blur.cpp
  src/cpu/sobel.cpp
  src/reference/opencv_gaussian_blur.cpp
  src/reference/opencv_sobel.cpp)
target_include_directories(gve_cpu PUBLIC include)
target_link_libraries(gve_cpu PUBLIC opencv_core opencv_imgproc)
target_compile_options(gve_cpu PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)

if(GVE_HAS_CUDA)
  add_library(gve_cuda src/cuda/gaussian_blur.cu src/cuda/sobel.cu)
  target_include_directories(gve_cuda PUBLIC include)
  target_link_libraries(gve_cuda PUBLIC CUDA::cudart opencv_core)
  set_target_properties(gve_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 75)
  target_compile_definitions(gve_cuda PUBLIC GVE_HAS_CUDA=1)
endif()

if(GVE_BUILD_BENCHMARKS)
  add_executable(gaussian_blur_benchmark benchmarks/gaussian_blur_benchmark.cpp benchmarks/environment.cpp)
  target_include_directories(gaussian_blur_benchmark PRIVATE include benchmarks)
  target_link_libraries(gaussian_blur_benchmark PRIVATE gve_cpu)
  target_compile_definitions(gaussian_blur_benchmark PRIVATE
    GVE_COMPILER_FLAGS="${CMAKE_CXX_FLAGS} $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}> $<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>")
  if(GVE_HAS_CUDA)
    target_link_libraries(gaussian_blur_benchmark PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(gaussian_blur_benchmark PRIVATE GVE_HAS_CUDA=1
      GVE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
      GVE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS} $<$<CONFIG:Release>:${CMAKE_CUDA_FLAGS_RELEASE}> $<$<CONFIG:Debug>:${CMAKE_CUDA_FLAGS_DEBUG}>")
  endif()

  add_executable(sobel_benchmark benchmarks/sobel_benchmark.cpp benchmarks/environment.cpp)
  target_include_directories(sobel_benchmark PRIVATE include benchmarks)
  target_link_libraries(sobel_benchmark PRIVATE gve_cpu)
  target_compile_definitions(sobel_benchmark PRIVATE
    GVE_COMPILER_FLAGS="${CMAKE_CXX_FLAGS} $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}> $<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>")
  if(GVE_HAS_CUDA)
    target_link_libraries(sobel_benchmark PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(sobel_benchmark PRIVATE GVE_HAS_CUDA=1
      GVE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
      GVE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS} $<$<CONFIG:Release>:${CMAKE_CUDA_FLAGS_RELEASE}> $<$<CONFIG:Debug>:${CMAKE_CUDA_FLAGS_DEBUG}>")
  endif()
endif()

if(GVE_BUILD_TESTS)
  include(CTest)
  include(FetchContent)
  FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_executable(gve_tests tests/gaussian_blur_test.cpp tests/sobel_test.cpp)
  target_link_libraries(gve_tests PRIVATE gve_cpu GTest::gtest_main)
  if(GVE_HAS_CUDA)
    target_link_libraries(gve_tests PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(gve_tests PRIVATE GVE_HAS_CUDA=1)
  endif()
  include(GoogleTest)
  gtest_discover_tests(gve_tests)
endif()
