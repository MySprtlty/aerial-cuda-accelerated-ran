# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
    add_definitions(-DENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
endif()

# Lib: scf_5g_fapi
set(SOURCES scf_5g_fapi.cpp
            scf_5g_fapi_mac.cpp
            scf_5g_fapi_phy.cpp
            scf_5g_fapi_module_dispatch.cpp
            scf_5g_slot_commands.cpp
            scf_5g_fapi_metrics.cpp
            scf_5g_fapi_ul_validate.cpp
            scf_5g_fapi_dl_validate.cpp
            scf_5g_slot_commands_common.cpp
            scf_5g_slot_commands_ssb_pbch.cpp
            scf_5g_slot_commands_pdcch.cpp
            scf_5g_slot_commands_pdsch_mod_comp.cpp
            scf_5g_slot_commands_pdsch_csirs.cpp
            scf_5g_fapi_dapp_export.cpp
            )

add_library(scf_5g_fapi STATIC ${SOURCES})
set_property(TARGET scf_5g_fapi PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(scf_5g_fapi PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(scf_5g_fapi PRIVATE cuphy_hdf5 cuphy_cpp_header_only nvphy nvipc slot_command)
if(ENABLE_DAPP_HOOK)
    target_link_libraries(scf_5g_fapi PRIVATE dapp_hook)
    target_compile_definitions(scf_5g_fapi PRIVATE ENABLE_DAPP_HOOK)
endif()

# Header-only INTERFACE library for SCF 5G FAPI headers
add_library(scf_5g_fapi_h INTERFACE)
add_library(scf::scf_5g_fapi_h ALIAS scf_5g_fapi_h)

target_include_directories(scf_5g_fapi_h INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(scf_5g_fapi_h INTERFACE cxx_std_20)

#---- Aerial metrics
if (TARGET aerial_metrics)
    target_link_libraries(scf_5g_fapi PRIVATE aerial_metrics)
    target_compile_definitions(scf_5g_fapi PRIVATE AERIAL_METRICS)
endif()

