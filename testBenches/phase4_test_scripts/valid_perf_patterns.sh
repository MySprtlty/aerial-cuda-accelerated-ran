#!/bin/bash
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

# Array of valid performance patterns
valid_perf_patterns=("ddddddsuuu_peak" "ddddddsuuu_l2" "ddddddsuuu_l3" "ddddddsuuu_l4" "ddddddsuuu_l5" "ddddddsuuu_peak_long" "ddddddsuuu_alt_long" "dsuuudsuuu_peak" "46" "47" "48" "49" "50" "51" "53" "54" "55" "56" "59" "59a" "59b" "59c" "59d" "59e" "59f" "60" "60a" "60b" "60c" "60d" "61" "62c" "63c" "65a" "65b" "65c" "65d" "66a" "66b" "66c" "66d" "67a" "67b" "67c" "67d" "67e" "69" "69a" "69b" "69c" "69d" "69e" "71" "73" "75" "77" "79" "81a" "81b" "81c" "81d" "83a" "83b" "83c" "83d" "85")

# Array of valid MIMO patterns
mimo_patterns=("66a" "66b" "66c" "66d" "67a" "67b" "67c" "67d" "67e" "69" "69a" "69b" "69c" "69d" "69e" "71" "73" "75" "77" "79" "81a" "81b" "81c" "81d" "83a" "83b" "83c" "83d" "85")

# Array of valid DFT-S-OFDM patterns
dft_s_ofdm_patterns=("65b" "65d" "66b" "66d" "67b" "67d")

# Array of peak patterns
peak_patterns=("ddddddsuuu_peak" "ddddddsuuu_l2" "ddddddsuuu_l3" "ddddddsuuu_l4" "ddddddsuuu_l5" "ddddddsuuu_peak_long" "ddddddsuuu_alt_long" "dsuuudsuuu_peak" "46" "48" "50" "53" "55" "59" "59a" "59b" "59c" "59d" "59e" "59f" "61" "62c" "65a" "65b" "65c" "65d" "67a" "67b" "67c" "67d" "67e" "69" "69a" "69b" "69c" "69d" "69e" "71" "73" "75" "77" "79" "81a" "81b" "81c" "81d" "83a" "83b" "83c" "83d" "85")

# Array of average patterns
average_patterns=("47" "49" "51" "54" "56" "60" "60a" "60b" "60c" "60d" "63c" "66a" "66b" "66c" "66d" "66e")

# Function to get the mode for a given pattern
get_pattern_mode() {
    local pattern=$1
    
    # Check if pattern is in peak_patterns
    if [[ " ${peak_patterns[*]} " =~ " $pattern " ]]; then
        echo "peak"
    elif [[ " ${average_patterns[*]} " =~ " $pattern " ]]; then
        echo "average"
    else
        echo "Pattern not found"
    fi
}