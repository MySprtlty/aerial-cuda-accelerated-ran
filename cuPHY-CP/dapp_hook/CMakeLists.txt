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

install(TARGETS dapp_ring_dump dapp_ring_fakel1 dapp_ring_selftest DESTINATION ${PROJECT_NAME}/bin OPTIONAL)
install(DIRECTORY include/ DESTINATION ${PROJECT_NAME}/include OPTIONAL)
