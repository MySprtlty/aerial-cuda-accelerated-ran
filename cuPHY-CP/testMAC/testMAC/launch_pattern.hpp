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

#ifndef _LAUNCH_PATTERN_HPP_
#define _LAUNCH_PATTERN_HPP_

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <unordered_map>

#include "nv_ipc_utils.h"
#include "cuphy.h"
#include "yaml.hpp"
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"

#include "fapi_defines.hpp"
#include "test_mac_configs.hpp"

// Dimensions: slot, cell, fapi_group, channel
using slot_pattern_t = std::vector<std::vector<std::vector<std::vector<fapi_req_t*>>>>;

using dbt_md_list_t = std::vector<dbt_md_t>;

class launch_pattern {
public:
    launch_pattern(test_mac_configs* configs);
    ~launch_pattern();
    int launch_pattern_parsing(const char* lp_file_name, uint32_t ch_mask, uint64_t cell_mask);
    int dynamic_pattern_parsing(yaml::node& slot_pattern, uint32_t ch_mask, uint64_t cell_mask);

    int save_harq_pid(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, uint8_t hpid);
    int read_harq_pid(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, uint8_t* hpid);

    cell_configs_t& get_cell_configs(int cell_id)
    {
        if(cell_id < cell_configs_v.size())
        {
            return *cell_configs_v[cell_id];
        }
        else
        {
            throw std::runtime_error("invalid cell_id");
        }
    }

    int get_fapi_type()
    {
        return fapi_type;
    }

    int get_cell_num()
    {
        return cell_num;
    }

    int get_slots_per_frame()
    {
        return slots_per_frame;
    }

    int get_sched_slot_num()
    {
        return sched_slot_num;
    }

    int get_negative_test()
    {
        return negative_test;
    }

    int get_config_static_harq_proc_id()
    {
        return config_static_harq_proc_id;
    }

    void set_config_static_harq_proc_id(int val)
    {
        config_static_harq_proc_id = val;
    }

    prach_configs_t& get_prach_configs(int cell_id)
    {
        return prach_reconfig_flag == 0 ? init_prach_configs[cell_id] : reconfig_prach_configs[cell_id];
    }

    int get_prach_reconfig_test()
    {
        return reconfig_slots_pattern.size() > 0 ? 1 : 0;
    }

    int get_mmimo_enabled()
    {
        return enable_dynamic_BF;
    }

    int get_mmimo_static_dynamic_enabled()
    {
        return enable_static_dynamic_BF;
    }

    int get_srs_enabled()
    {
        return enable_srs;
    }

    std::vector<precoding_matrix_t>& get_precoding_matrix_v(int cell_id)
    {
        return precoding_matrix_v[cell_id];
    }

    slot_pattern_t& get_slot_cell_patterns()
    {
        return using_init_patterns ? init_slots_pattern : sched_slots_pattern;
    }

    slot_pattern_t& get_slot_cell_patterns(uint64_t slot_counter)
    {
        if (yaml_configs->app_mode != 0)
        {
            return dynamic_slots_pattern;
        }

        if(slot_counter < init_slot_num)
        {
            return init_slots_pattern;
        }
        else
        {
            return sched_slots_pattern;
        }
    }
 
    dyn_slot_param_t& get_dyn_slot_param(int32_t cell_id, sfn_slot_t& ss)
    {
        return dynamic_slot_params.at(ss.u16.slot).at(cell_id);
    }

    static_slot_param_t& get_static_slot_param(int32_t cell_id)
    {
        return static_slot_params.at(cell_id);
    }

    slot_pattern_t& get_sched_slots_patterns()
    {
        return sched_slots_pattern;
    }

    void check_init_pattern_finishing(uint64_t scheduled_slot_num)
    {
        if (using_init_patterns == false)
        {
            return;
        }

        if (scheduled_slot_num == init_slot_num)
        {
            using_init_patterns = false;
        }
    }

    int apply_reconfig_pattern(int cell_id);

    int populate_launch_pattern(slot_pattern_t& slots_data, yaml::node cell_config, int cell_id, int slot_idx);
    cv_membank_config_req_t& get_mem_bank_configs(int cell_id) {
        return mem_bank_config_req[cell_id];
    }

    // night-3 test option (2026-10-18), default OFF = the original (sfn * slots_per_frame + slot) % pattern size.
    // TESTMAC_LP_BLOCK_REPEAT=<N> (+ TESTMAC_LP_BLOCK_FRAMES, default 4): the pattern file's frames form blocks of
    // TESTMAC_LP_BLOCK_FRAMES frames, block k is replayed for N consecutive real blocks, then block k+1 (wrapping). The real
    // frame counter is extended beyond SFN 1023 with CLOCK_REALTIME (frame 0 of SFN 0 at t = 3.64 s mod 10.24 s), nearest-match
    // adjusted to the slot's SFN. The RU emulator implements the same mapping (RU_LP_BLOCK_REPEAT), so both pick the same vectors.
    uint32_t lp_pattern_index(uint16_t sfn, uint16_t slot, size_t pattern_slots)
    {
        static const int rep = [] { const char* e = getenv("TESTMAC_LP_BLOCK_REPEAT"); return e ? atoi(e) : 0; }();
        if (rep <= 0 || pattern_slots == 0 || slots_per_frame == 0)
            return get_slot_in_frame(sfn, slot) % pattern_slots;
        static const int blk = [] { const char* e = getenv("TESTMAC_LP_BLOCK_FRAMES"); const int n = e ? atoi(e) : 4; return n > 0 ? n : 4; }();
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        const int64_t t_ns   = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
        const int64_t f_wall = (t_ns - 3640000000LL) / 10000000LL;
        int64_t d = ((int64_t)sfn - f_wall) % 1024;
        if (d < 0) d += 1024;
        if (d > 512) d -= 1024;
        const int64_t ext      = f_wall + d;
        const int     n_frames = (int)(pattern_slots / slots_per_frame);
        const int     n_blocks = (n_frames / blk) > 0 ? (n_frames / blk) : 1;
        const int64_t seg      = (ext / ((int64_t)blk * rep)) % n_blocks;
        const int     fib      = (int)(ext % blk);
        return (uint32_t)(((seg * blk + fib) * slots_per_frame + slot) % pattern_slots);
    }

    uint32_t get_slot_in_frame(uint16_t sfn, uint16_t slot)
    {
        uint32_t index = sfn; // extend to uint32_t
        return index * slots_per_frame + slot;
    }

    std::vector<channel_segment_t>&  get_channel_segment(int32_t cell_id) {
        return ch_segment_map[cell_id];
    }


    dbt_md_t* get_dbt_info(int cell_id)
    {
        auto size = dbt_per_cell_md_list.size();
        if (!size || size < cell_id) {
            return nullptr;
        }
        return &dbt_per_cell_md_list[cell_id];
    }

    csi2_maps_t* get_csi2_maps(int cell_id) {
        
        auto size = csi2maps.size();
        if (!size || size < cell_id) {
            return nullptr;
        }

        return &csi2maps[cell_id];
    }

    std::vector<int32_t>& get_lp_cell_id_vec()
    {
        return lp_cell_id_vec;
    }

    std::vector<thrput_t>& get_expected_values() 
    {
        return expected; 
    }

private:
    int parse_tv_pdcch(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_pdsch(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_pucch(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_pusch(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_pbch(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_prach(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_csirs(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_srs(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_bfw(hdf5hpp::hdf5_file& file, fapi_req_t* req);
    int parse_tv_cv_membank_configs(hdf5hpp::hdf5_file& file, fapi_req_t* req);

    int parse_precoding_matrix(hdf5hpp::hdf5_file& hdf5file, int cell_id);
    int parse_tx_beamforming(hdf5hpp::hdf5_dataset& hdf5dset, tx_beamforming_data_t& beam_data);
    int parse_rx_beamforming(hdf5hpp::hdf5_dataset& hdf5dset, rx_beamforming_data_t& beam_data);
    int parse_srs_rx_beamforming(hdf5hpp::hdf5_dataset& hdf5dset, rx_srs_beamforming_data_t& beam_data);
    int parse_beam_ids(yaml::node map_node);
    int set_beam_ids(uint8_t& digBFInterfaces, std::vector<uint16_t>& beamIdx_v);

    int parse_tv_channels(slot_pattern_t& slots_data, int cell_id, int slot_id, int ch_mask, std::string tv_file, std::vector<std::string>& channel_type_list);
    int parse_tv_file(fapi_req_t* req);
    int parse_pdu_dset_names(hdf5hpp::hdf5_file& file, tv_dset_list& list);
    int load_h5_config_params(int cell_id, const char* config_params_h5_file, const char* ul_params_h5_file);
    int update_expected_values(int cell_id, fapi_req_t* req);

    int init_prach_config(int cell_num);
    int parse_prach_config(hdf5hpp::hdf5_file& file, int cell_id);

    void parse_slots(yaml::node& slot_list, slot_pattern_t& slots_data);
    int parse_time_lines(hdf5hpp::hdf5_file& file, int cell_id);
    int parse_dbt_configs(hdf5hpp::hdf5_file& file, int cell_id);

    int add_dyn_slot_channel(slot_pattern_t& slots_data, int cell_id, int slot_id, channel_type_t ch);
    void read_csip2_maps(hdf5hpp::hdf5_file & file, int cell_id);

private:
    int fapi_type;
    int channel_mask;

    std::vector<int32_t> lp_cell_id_vec; // map<cell_id, lp_cell_id>

    int cell_num;
    int sched_slot_num;
    int init_slot_num;
    int slots_per_frame;
    int config_static_harq_proc_id;
    int prach_reconfig_flag;

    bool using_init_patterns;

    // TestMAC YAML configs
    test_mac_configs* yaml_configs;

    // PRACH configurations
    std::vector<prach_configs_t> init_prach_configs;
    std::vector<prach_configs_t> reconfig_prach_configs;

    // Expected throughput data parsed from TV
    std::vector<thrput_t> expected;

    // Dimensions: slot, cell, fapi_group, channel
    slot_pattern_t init_slots_pattern;
    slot_pattern_t sched_slots_pattern;
    slot_pattern_t reconfig_slots_pattern;

    slot_pattern_t dynamic_slots_pattern;

    // Dimensions: slot, cell
    std::vector<std::vector<dyn_slot_param_t>> dynamic_slot_params;
    std::vector<static_slot_param_t> static_slot_params;

    // <channel, <tv_name, tv_file>*>
    std::unordered_map<int, std::unordered_map<std::string, std::string>*> tv_maps;

    std::vector<cell_configs_t*> cell_configs_v;

    std::vector<uint16_t> global_beam_idx_v;
    std::vector<uint16_t> current_beam_idx_v;

    // PrecodingMatrix
    std::vector<std::vector<precoding_matrix_t>> precoding_matrix_v;

    // negative test flag
    int negative_test;

    // 32T32R Beamforming flag
    int enable_dynamic_BF;

    // 64T64R static+dynamic Beamforming flag
    int enable_static_dynamic_BF;

    // Flag to enable SRS
    int enable_srs;

    // Save harqProcessID values for validation
    std::vector<std::unordered_map<uint32_t, std::vector<uint8_t>>> hpid_maps;

    // TV data map: unordered_map<filename, test_vector_t*> per channel
    std::array<std::unordered_map<std::string, test_vector_t*>, CHANNEL_MAX> tv_data_maps;

    std::unordered_map<std::string, hdf5hpp::hdf5_file> h5file_map;

    cv_membank_config_req_t mem_bank_config_req[PDSCH_MAX_CELLS_PER_CELL_GROUP];

    std::unordered_map<int32_t, std::vector<channel_segment_t>> ch_segment_map;

    dbt_md_list_t dbt_per_cell_md_list;

    std::vector<csi2_maps_t> csi2maps;

    std::array<bool, PDSCH_MAX_CELLS_PER_CELL_GROUP> cv_membank_config_read_dl_bfw;
    std::array<bool, PDSCH_MAX_CELLS_PER_CELL_GROUP> cv_membank_config_read_ul_bfw;
};

#endif /* _LAUNCH_PATTERN_HPP_ */
