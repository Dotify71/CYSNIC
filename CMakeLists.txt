cmake_minimum_required(VERSION 3.10)
project(CYSNIC VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG        v2021.11.0
)
FetchContent_MakeAvailable(tbb)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(cysnic_tracker 
    src/main.cpp
    src/tracker.cpp
)

target_link_libraries(cysnic_tracker ${OpenCV_LIBS} Eigen3::Eigen TBB::tbb)
