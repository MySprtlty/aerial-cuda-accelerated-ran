# xapp_flexric: a FlexRIC xApp that sends RUN requests to the DGIST dApp agent (see README.md)
#
#   cmake -S . -B build -DFLEXRIC_DIR=/path/to/flexric      # the FlexRIC source tree that was built (needs <dir>/build)
#   cmake --build build
#   -> build/xapp_dapp_run        (FlexRIC + dApp)        only when FLEXRIC_DIR/build/src/xApp/libe42_xapp.a exists
#   -> build/xapp_dapp_run_noric  (dApp path only, no FlexRIC dependency)
cmake_minimum_required(VERSION 3.16)
project(xapp_dapp_run C)

set(FLEXRIC_DIR "/home/nvidia/hsw/25_3_hsw/openairinterface5g/openair2/E2AP/flexric" CACHE PATH "FlexRIC source tree (the one the gNB's E2 agent and the nearRT-RIC were built from)")
set(FLEXRIC_BUILD "${FLEXRIC_DIR}/build" CACHE PATH "its build directory")
# must match the FlexRIC build (build/CMakeCache.txt: E2AP_VERSION, KPM_VERSION)
set(E2AP_VERSION "E2AP_V2" CACHE STRING "E2AP_V1 | E2AP_V2 | E2AP_V3")
set(KPM_VERSION "KPM_V2_03" CACHE STRING "KPM_V2_01 | KPM_V2_03 | KPM_V3_00")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-function -Wno-unused-parameter)

find_package(Threads REQUIRED)

# 1. dApp path only (protocol test, no FlexRIC)
add_executable(xapp_dapp_run_noric xapp_dapp_run.c)
target_compile_definitions(xapp_dapp_run_noric PRIVATE NO_FLEXRIC)
target_link_libraries(xapp_dapp_run_noric PRIVATE Threads::Threads m)

# 2. the FlexRIC xApp
set(E42_LIB "${FLEXRIC_BUILD}/src/xApp/libe42_xapp.a")
if(EXISTS "${E42_LIB}")
  message(STATUS "FlexRIC xApp library: ${E42_LIB} (${E2AP_VERSION}, ${KPM_VERSION})")
  add_executable(xapp_dapp_run xapp_dapp_run.c)
  target_include_directories(xapp_dapp_run PRIVATE "${FLEXRIC_DIR}/src")
  target_compile_definitions(xapp_dapp_run PRIVATE ASN ${E2AP_VERSION} ${KPM_VERSION} SQLITE3_XAPP)
  find_library(SQLITE3_LIB sqlite3)
  set(EXTRA_LIBS "")
  if(SQLITE3_LIB)
    set(EXTRA_LIBS ${SQLITE3_LIB})
  endif()
  # libe42_xapp's sqlite wrapper decodes RRC messages (RC SM): the ASN.1 RRC archive built with the FlexRIC examples
  file(GLOB RRC_ASN1_LIB "${FLEXRIC_BUILD}/examples/xApp/c/monitor/RRC_MESSAGES/libasn1_nr_rrc.a" "${FLEXRIC_BUILD}/examples/xApp/c/*/RRC_MESSAGES/libasn1_nr_rrc.a")
  if(RRC_ASN1_LIB)
    list(GET RRC_ASN1_LIB 0 RRC_ASN1_LIB)
    message(STATUS "RRC ASN.1 library: ${RRC_ASN1_LIB}")
    list(APPEND EXTRA_LIBS "${RRC_ASN1_LIB}")
  endif()
  target_link_libraries(xapp_dapp_run PRIVATE "${E42_LIB}" Threads::Threads sctp dl m ${EXTRA_LIBS})
else()
  message(WARNING "FlexRIC build not found at ${E42_LIB}: only xapp_dapp_run_noric is built. Set -DFLEXRIC_DIR=<flexric source tree with build/>.")
endif()
