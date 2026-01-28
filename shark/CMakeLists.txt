cmake_minimum_required(VERSION 3.16)
project(shark)

enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (Eigen3 REQUIRED NO_MODULE)

add_subdirectory(ext/cryptoTools)

add_library(${PROJECT_NAME}
    src/utils/timer.cpp
    src/utils/comm.cpp
    src/protocols/common.cpp
    src/protocols/init.cpp
    src/protocols/finalize.cpp
    src/protocols/input.cpp
    src/protocols/output.cpp
    src/protocols/matmul.cpp
    src/protocols/relutruncate.cpp
    src/protocols/drelu.cpp
    src/protocols/select.cpp
    src/protocols/relu.cpp
    src/protocols/conv.cpp
    src/protocols/maxpool.cpp
    src/protocols/spline.cpp
    src/protocols/lrs.cpp
    src/protocols/sumpool.cpp
    src/protocols/mul.cpp
    src/protocols/lut.cpp
    src/protocols/reciprocal.cpp
    src/crypto/dcfbit.cpp
    src/crypto/dcfring.cpp
    src/crypto/dpfring.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen cryptoTools)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# list of all single running tests
set(SINGLE_TESTS
    dcfbit
    dcfring
    dpfring
    span
)

# run for all single running tests
foreach(testname IN LISTS SINGLE_TESTS)
    add_executable(test-single-${testname} tests/single/${testname}.cpp)
    target_link_libraries(test-single-${testname} ${PROJECT_NAME})
    add_test(NAME test-single-${testname} COMMAND test-single-${testname})
endforeach()

set(MULTI_TESTS
    init
    matmul
    relutruncate
    select
    matmul-big
    drelu
    relu
    # conv # failing
    maxpool
    spline
    sigmoid
    tanh
    # reciprocal # failing
    lut
)

foreach(testname IN LISTS MULTI_TESTS)
    add_executable(test-multi-${testname} tests/multi/${testname}.cpp)
    target_link_libraries(test-multi-${testname} ${PROJECT_NAME})
    add_test(NAME test-multi-${testname} COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/run-multi.sh multi-${testname})
endforeach()

set (BENCHMARKS
    deepsecure1
    deepsecure2
    deepsecure3
    deepsecure4
    simc1
    simc2
    bert
    alexnet
    alexnet-localtrunc
    mnist-A
    hinet
    vgg16
)

foreach (benchmark IN LISTS BENCHMARKS)
    add_executable(benchmark-${benchmark} benchmarks/${benchmark}.cpp)
    target_link_libraries(benchmark-${benchmark} ${PROJECT_NAME})
endforeach()

# CROWN benchmarks (in subdirectory)
add_executable(crowntest benchmarks/crown/crowntest.cpp)
target_link_libraries(crowntest ${PROJECT_NAME})

add_executable(crowntest_optimized benchmarks/crown/crowntest_optimized.cpp)
target_link_libraries(crowntest_optimized ${PROJECT_NAME})

add_executable(crowntest_batch benchmarks/crown/crowntest_batch.cpp)
target_link_libraries(crowntest_batch ${PROJECT_NAME})

add_executable(crowntest_batch_vec benchmarks/crown/crowntest_batch_vec.cpp)
target_link_libraries(crowntest_batch_vec ${PROJECT_NAME})

add_executable(crowntest_batch_true benchmarks/crown/crowntest_batch_true.cpp)
target_link_libraries(crowntest_batch_true ${PROJECT_NAME})

add_executable(crowntest_v2 benchmarks/crown/crowntest_v2.cpp)
target_link_libraries(crowntest_v2 ${PROJECT_NAME})

add_executable(crowntest_v3 benchmarks/crown/crowntest_v3.cpp)
target_link_libraries(crowntest_v3 ${PROJECT_NAME})

set (MICROBENCHMARKS
    relu
    spline
    relutruncate
    maxpool
    reciprocal
    matmul
    linear
)

foreach (microbenchmark IN LISTS MICROBENCHMARKS)
    add_executable(micro-${microbenchmark} microbenchmarks/${microbenchmark}.cpp)
    target_link_libraries(micro-${microbenchmark} ${PROJECT_NAME})
endforeach()
