cmake_minimum_required(VERSION 3.16)
project(asl)

# Set build type to RELEASE by default:
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

# Options can be turned on and off by providing cmake with a -DOPTIONNAME=ON (or OFF) argument.
# For example, to turn off benchmarks, build a shared library, and use sanitize flags in a DEBUG build:
#    cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DASL_BENCH=OFF -DBENCH_STATIC=OFF -DASL_SANITIZE=ON
option(ASL_TEST      "Build tests" ON)
option(ASL_BENCH     "Build benchmarks" ON)
option(ASL_STATIC    "Builds a static library instead of a shared one" OFF)
# Adds extra wrappers around memory accesses to detect memory errors (out of bounds, use after free)
option(ASL_SANITIZE  "Adds sanitive flags" OFF)

if(ASL_STATIC)
  set(ASL_LIB_NAME asl_static)
  set(ASL_LIB_TYPE STATIC)
else()
  set(ASL_LIB_NAME asl)
  set(ASL_LIB_TYPE SHARED)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(FetchContent)

include_directories(. include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Verbose and pedantic error messages
set(CMAKE_CXX_FLAGS "-Wall -Werror -pedantic ${CMAKE_CXX_FLAGS}")

if(ASL_SANITIZE)
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(src)

if(ASL_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

if(ASL_BENCH)
  add_subdirectory(bench)
endif()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library name         : ${ASL_LIB_NAME}")
message(STATUS "  Library type         : ${ASL_LIB_TYPE}")
message(STATUS "  Build tests          : ${ASL_TEST}")
message(STATUS "  Build benchmarks     : ${ASL_BENCH}")
message(STATUS "  Sanitize flags       : ${ASL_SANITIZE}")
message(STATUS "")
