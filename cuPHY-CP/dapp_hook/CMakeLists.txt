# dApp FAPI hook: shared-memory ring exported by the L1 (cuphycontroller)
# for external dApps (GPU schedulers, analytics, ...).
#
#   dapp_hook          header-only library (producer/consumer, ABI)
#   dapp_ring_dump     CLI that prints the ring contents (verification)
#   dapp_ring_selftest producer/consumer integrity test, no L1 required

add_library(dapp_hook INTERFACE)
target_include_directories(dapp_hook INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(dapp_hook INTERFACE cxx_std_17)
if(ENABLE_DAPP_HOOK)
    target_compile_definitions(dapp_hook INTERFACE ENABLE_DAPP_HOOK)
endif()

find_library(DAPP_LIBRT rt)
if(DAPP_LIBRT)
    target_link_libraries(dapp_hook INTERFACE ${DAPP_LIBRT})
endif()

add_executable(dapp_ring_dump tools/dapp_ring_dump.cpp)
target_link_libraries(dapp_ring_dump PRIVATE dapp_hook)
set_target_properties(dapp_ring_dump PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_executable(dapp_ring_fakel1 tools/dapp_ring_fakel1.cpp)
target_link_libraries(dapp_ring_fakel1 PRIVATE dapp_hook)
set_target_properties(dapp_ring_fakel1 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

add_executable(dapp_ring_selftest tools/dapp_ring_selftest.cpp)
target_link_libraries(dapp_ring_selftest PRIVATE dapp_hook pthread)
set_target_properties(dapp_ring_selftest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# dapp_sched: rule-based SM budget scheduler prototype. Needs the CUDA driver
# API; skipped when that target is unavailable.
if(TARGET CUDA::cuda_driver)
    add_executable(dapp_sched sched/dapp_sched_main.cpp)
    target_include_directories(dapp_sched PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sched)
    target_link_libraries(dapp_sched PRIVATE dapp_hook CUDA::cuda_driver pthread)
    set_target_properties(dapp_sched PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    install(TARGETS dapp_sched DESTINATION ${PROJECT_NAME}/bin OPTIONAL)

    # Real inference: a YOLO engine through TensorRT, one instance per SM class
    # (sched/dapp_yolo.hpp). Optional; without TensorRT the placeholder kernel
    # is all dapp_sched can run. Image decoding is the vendored stb_image,
    # compiled on its own with warnings off so -Werror does not apply to it.
    find_library(DAPP_NVINFER_LIB nvinfer)
    find_path(DAPP_NVINFER_INC NvInfer.h PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
    if(DAPP_NVINFER_LIB AND DAPP_NVINFER_INC AND TARGET CUDA::cudart)
        target_sources(dapp_sched PRIVATE third_party/stb_image_impl.cpp)
        set_source_files_properties(third_party/stb_image_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
        target_compile_definitions(dapp_sched PRIVATE DAPP_SCHED_TENSORRT=1)
        target_include_directories(dapp_sched PRIVATE ${DAPP_NVINFER_INC} ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
        target_link_libraries(dapp_sched PRIVATE ${DAPP_NVINFER_LIB} CUDA::cudart)
        message(STATUS "dapp_sched: TensorRT YOLO runner enabled (${DAPP_NVINFER_LIB})")
    else()
        message(STATUS "dapp_sched: TensorRT not found, YOLO runner disabled")
    endif()
else()
    message(STATUS "dapp_sched skipped: CUDA::cuda_driver target not available")
endif()

# dapp_export_test: builds synthetic SCF FAPI DL_TTI/UL_TTI messages in memory,
# runs the same export code the L1 hook uses, and checks the records. Compiles
# the export source directly so it needs no L1 library; the FAPI defines must
# match the scf_5g_fapi build or the struct layouts diverge.
if(ENABLE_DAPP_HOOK)
    add_executable(dapp_export_test tools/dapp_export_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_dapp_export.cpp)
    target_include_directories(dapp_export_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../scfl2adapter/lib/scf_5g_fapi
        ${CMAKE_CURRENT_SOURCE_DIR}/../gt_common_libs/nvIPC/include)
    target_link_libraries(dapp_export_test PRIVATE dapp_hook)
    target_compile_definitions(dapp_export_test PRIVATE ENABLE_DAPP_HOOK)
    if(ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
        target_compile_definitions(dapp_export_test PRIVATE ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
    endif()
    set_target_properties(dapp_export_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endif()

install(TARGETS dapp_ring_dump dapp_ring_fakel1 dapp_ring_selftest DESTINATION ${PROJECT_NAME}/bin OPTIONAL)
install(DIRECTORY include/ DESTINATION ${PROJECT_NAME}/include OPTIONAL)
