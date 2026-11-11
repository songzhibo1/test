set(CROWN_SOURCES
    crown_ops.cpp
    crown_standard.cpp
    crown_standard_kernel.cpp)

add_library(crown_core STATIC ${CROWN_SOURCES})
target_include_directories(crown_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/external/EzPC/SCI/src)

if(CROWN_BACKEND STREQUAL "HE")
  target_link_libraries(crown_core PUBLIC SCI-HE)
  target_compile_definitions(crown_core PRIVATE SCI_HE=1)
elseif(CROWN_BACKEND STREQUAL "OT")
  target_link_libraries(crown_core PUBLIC SCI-OT)
  target_compile_definitions(crown_core PRIVATE SCI_OT=1)
else()
  message(FATAL_ERROR "Invalid CROWN_BACKEND='${CROWN_BACKEND}', expected OT or HE.")
endif()

add_library(crown_plaintext STATIC crown_plaintext.cpp)
target_include_directories(crown_plaintext
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cryptflow2_crown main.cpp crown_mlp_file.cpp)
target_link_libraries(cryptflow2_crown PRIVATE crown_core crown_plaintext)
target_include_directories(cryptflow2_crown PRIVATE ${CMAKE_SOURCE_DIR}/external/EzPC/SCI/src)

if(CROWN_BACKEND STREQUAL "HE")
  target_compile_definitions(cryptflow2_crown PRIVATE SCI_HE=1)
else()
  target_compile_definitions(cryptflow2_crown PRIVATE SCI_OT=1)
endif()

if(CROWN_ENABLE_SMOKE_TEST)
  enable_testing()
  add_test(
    NAME cryptflow2_crown_localhost
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_localhost_check.sh $<TARGET_FILE:cryptflow2_crown>)
endif()
