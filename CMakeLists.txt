cmake_minimum_required(VERSION 3.10)
project(CYSNIC VERSION 1.0.0 LANGUAGES CXX)

option(USE_SANITIZERS "Enable Address and Undefined Behavior Sanitizers" OFF)
option(USE_CLANG_TIDY "Enable Clang-Tidy Static Analysis" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3147391d946bb4b6c68edd901f2add6ac1f31f8c # 3.4.0
)
FetchContent_MakeAvailable(eigen)

set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG        8b829acc65569019edb896c5150d427f288e8aba # v2021.11.0
)
FetchContent_MakeAvailable(tbb)

# FFTW requires CMake 3.0+ but it works via FetchContent if we disable tests
set(ENABLE_THREADS ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  fftw
  GIT_REPOSITORY https://github.com/FFTW/fftw3.git
  GIT_TAG        7184fc796279cfa70e4ba62519ac2938054584e6 # fftw-3.3.10
)
FetchContent_MakeAvailable(fftw)

# spdlog for tactical logging
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        7e635fca68d014934b4af8a1cf874f63989352b7 # v1.12.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.2 # Explicit version tag for stability
)
FetchContent_MakeAvailable(nlohmann_json)

# GoogleTest for KF math validation
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        f8d7d77c06936315286eb55f8de22cd23c188571 # v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${OpenCV_INCLUDE_DIRS} include ${fftw_SOURCE_DIR}/api)

# Default compiler flags for rigorous build
set(CYSNIC_FLAGS "-Wall" "-Wextra" "-Wpedantic" "-Werror")

if(USE_SANITIZERS)
    list(APPEND CYSNIC_FLAGS "-fsanitize=address,undefined" "-fno-omit-frame-pointer" "-g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined")
endif()

if(USE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE} "-checks=*,-fuchsia-*,-google-*,-zircon-*")
    else()
        message(WARNING "clang-tidy requested but not found!")
    endif()
endif()

# Append O3 after potential sanitizer flags
list(APPEND CYSNIC_FLAGS "-O3")

add_executable(cysnic_tracker 
    src/main.cpp 
    src/tracker.cpp
)

target_compile_options(cysnic_tracker PRIVATE ${CYSNIC_FLAGS})
target_link_libraries(cysnic_tracker PUBLIC ${OpenCV_LIBS} Eigen3::Eigen fftw3 spdlog::spdlog TBB::tbb nlohmann_json::nlohmann_json)

# Tests
enable_testing()
add_executable(cysnic_tests 
    tests/test_kf.cpp
    tests/test_main.cpp
    src/tracker.cpp
)
target_compile_options(cysnic_tests PRIVATE ${CYSNIC_FLAGS})
target_link_libraries(cysnic_tests PRIVATE gtest_main ${OpenCV_LIBS} Eigen3::Eigen fftw3 spdlog::spdlog TBB::tbb nlohmann_json::nlohmann_json)
include(GoogleTest)
gtest_discover_tests(cysnic_tests)
