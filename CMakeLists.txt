cmake_minimum_required(VERSION 3.24)
project(gpu_vision_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GVE_ENABLE_CUDA "Enable CUDA automatically when a compiler is available" ON)
option(GVE_BUILD_TESTS "Build tests" ON)
option(GVE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(GVE_BUILD_VIEWER "Build the optional OpenGL nuScenes viewer" OFF)

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
  src/cpu/resize.cpp
  src/cpu/sobel.cpp
  src/cpu/voxelization.cpp
  src/reference/opencv_gaussian_blur.cpp
  src/reference/opencv_resize.cpp
  src/reference/opencv_sobel.cpp)
target_include_directories(gve_cpu PUBLIC include)
target_link_libraries(gve_cpu PUBLIC opencv_core opencv_imgproc)
target_compile_options(gve_cpu PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)

if(GVE_HAS_CUDA)
  add_library(gve_cuda
    src/cuda/gaussian_blur.cu
    src/cuda/resize.cu
    src/cuda/sobel.cu
    src/cuda/voxelization.cu)
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

  add_executable(resize_benchmark benchmarks/resize_benchmark.cpp benchmarks/environment.cpp)
  target_include_directories(resize_benchmark PRIVATE include benchmarks)
  target_link_libraries(resize_benchmark PRIVATE gve_cpu)
  target_compile_definitions(resize_benchmark PRIVATE
    GVE_COMPILER_FLAGS="${CMAKE_CXX_FLAGS} $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}> $<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>")
  if(GVE_HAS_CUDA)
    target_link_libraries(resize_benchmark PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(resize_benchmark PRIVATE GVE_HAS_CUDA=1
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

  add_executable(voxelization_benchmark
    benchmarks/voxelization_benchmark.cpp
    benchmarks/environment.cpp)
  target_include_directories(voxelization_benchmark PRIVATE include benchmarks)
  target_link_libraries(voxelization_benchmark PRIVATE gve_cpu)
  target_compile_definitions(voxelization_benchmark PRIVATE
    GVE_COMPILER_FLAGS="${CMAKE_CXX_FLAGS} $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}> $<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>")
  if(GVE_HAS_CUDA)
    target_link_libraries(voxelization_benchmark PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(voxelization_benchmark PRIVATE GVE_HAS_CUDA=1
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
  add_executable(gve_tests
    tests/gaussian_blur_test.cpp
    tests/resize_test.cpp
    tests/sobel_test.cpp
    tests/voxelization_test.cpp)
  target_link_libraries(gve_tests PRIVATE gve_cpu GTest::gtest_main)
  if(GVE_HAS_CUDA)
    target_link_libraries(gve_tests PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(gve_tests PRIVATE GVE_HAS_CUDA=1)
  endif()
  include(GoogleTest)
  gtest_discover_tests(gve_tests)
endif()

if(GVE_BUILD_VIEWER)
  find_package(OpenGL REQUIRED)
  include(FetchContent)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(glfw)
  add_executable(voxel_viewer viewer/voxel_viewer.cpp)
  target_include_directories(voxel_viewer PRIVATE include)
  target_link_libraries(voxel_viewer PRIVATE gve_cpu glfw OpenGL::GL)
  if(GVE_HAS_CUDA)
    target_link_libraries(voxel_viewer PRIVATE gve_cuda CUDA::cudart)
    target_compile_definitions(voxel_viewer PRIVATE GVE_HAS_CUDA=1)
  endif()
  if(APPLE)
    target_compile_definitions(voxel_viewer PRIVATE GL_SILENCE_DEPRECATION)
  endif()
  target_compile_options(voxel_viewer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)
endif()
