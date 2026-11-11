cmake_minimum_required(VERSION 3.16)

project(cryptflow2_crown LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CROWN_BACKEND "OT" CACHE STRING "SCI backend for CROWN (OT or HE)")
set_property(CACHE CROWN_BACKEND PROPERTY STRINGS OT HE)
option(CROWN_ENABLE_SMOKE_TEST "Enable localhost 2PC smoke test in CTest" OFF)

set(SCI_ROOT "${CMAKE_SOURCE_DIR}/external/EzPC/SCI")
if(NOT EXISTS "${SCI_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
          "SCI source not found at ${SCI_ROOT}. Run scripts/bootstrap.sh first.")
endif()

# Keep SCI as a dependency-only build here.
set(BUILD_TESTS OFF CACHE BOOL "Build SCI tests" FORCE)
set(BUILD_NETWORKS OFF CACHE BOOL "Build SCI networks" FORCE)

add_subdirectory("${SCI_ROOT}" external/SCI)
add_subdirectory(src)

