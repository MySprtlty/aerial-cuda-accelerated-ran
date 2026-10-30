/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <utility>

namespace nv {

    struct phy_config_option
    {
        /* data */
        bool precoding_enabled;
        bool bf_enabled;
        int enableTickDynamicSfnSlot;
        int staticPuschSlotNum;
        int staticPdschSlotNum;
        int staticPdcchSlotNum;
        int staticCsiRsSlotNum;
        int staticSsbPcid;
        int staticSsbSFN;
        int staticSsbSlotNum;
        int staticPucchSlotNum;
        bool duplicateConfigAllCells;
        // Hybrid L1 (2026-10-29): the static*SlotNum / staticSsb* overrides are test-vector helpers (F08 vectors are generated for slot 0) and apply
        // to every cell. 'staticSlotCellMask' (L2 adapter yaml, bit c = L1 cell c; default all cells = upstream behaviour) limits them to the
        // emulator cells so that a real-UE cell in the same cell group keeps its real slot numbers / dynamic SFN.
        uint64_t staticSlotCellMask;
        bool staticSlotCellMaskSet;
        bool staticCell(int cell) const { return cell >= 0 && cell < 64 && ((staticSlotCellMask >> cell) & 1ULL); }
        int staticSlot(int static_value, int cell) const { return staticCell(cell) ? static_value : -1; }
        bool staticSfnForCell(int cell) const { return staticSlotCellMaskSet && staticCell(cell); }   // mask given: static-slot cells also keep the static SSB SFN
        explicit phy_config_option():
        precoding_enabled(false),
        bf_enabled(false),
        enableTickDynamicSfnSlot(1),
        staticPuschSlotNum(-1),
        staticPdschSlotNum(-1),
        staticPdcchSlotNum(-1),
        staticCsiRsSlotNum(-1),
        staticSsbPcid(-1),
        staticSsbSFN(-1),
        staticSsbSlotNum(-1),
        staticPucchSlotNum(-1),
        duplicateConfigAllCells(false),
        staticSlotCellMask(~0ULL),
        staticSlotCellMaskSet(false)
        {}

        phy_config_option(phy_config_option&& other):
        precoding_enabled(std::move(other.precoding_enabled)),
        bf_enabled(std::move(other.bf_enabled)),
        enableTickDynamicSfnSlot(std::move(other.enableTickDynamicSfnSlot)),
        staticPuschSlotNum(std::move(other.staticPuschSlotNum)),
        staticPdschSlotNum(std::move(other.staticPdschSlotNum)),
        staticPdcchSlotNum(std::move(other.staticPdcchSlotNum)),
        staticCsiRsSlotNum(std::move(other.staticCsiRsSlotNum)),
        staticSsbPcid(std::move(other.staticSsbPcid)),
        staticSsbSFN(std::move(other.staticSsbSFN)),
        staticSsbSlotNum(std::move(other.staticSsbSlotNum)),
        staticPucchSlotNum(std::move(other.staticPucchSlotNum)),
        duplicateConfigAllCells(std::move(other.duplicateConfigAllCells)),
        staticSlotCellMask(other.staticSlotCellMask),
        staticSlotCellMaskSet(other.staticSlotCellMaskSet)
        {}

        phy_config_option(phy_config_option&) = delete;
        phy_config_option& operator=(phy_config_option&) = delete;

    };

}