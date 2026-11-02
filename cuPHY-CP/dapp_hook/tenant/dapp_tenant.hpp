/*
 * dapp_tenant.hpp (2026-11-01): the YOLO tenant inside the cuphycontroller process.
 *
 * A segmented TensorRT network (tools/seg_yolo: models/seg26n_fine, one engine per segment and SM class) is run on CUDA
 * green contexts created in this process from the same device resource splits cuphydriver uses for its channel
 * partitions, so that every tenant partition is disjoint from the PUSCH / PDSCH partitions active in the current slot:
 *   gap(P, N) = remainder of split(prefix(P), N), prefix(P) = result of the PUSCH split (P = 132 - Y'), N = PDSCH profile
 *   (verified in tools/gc_bench/gc_nest.cu); without PUSCH, gap(N) = remainder of split(all, N).
 * The sizer (include/dapp_hook/dapp_sizer.hpp, running on the L2 adapter thread) records per slot the PUSCH / PDSCH
 * profiles it chose and the slot's t0; the tenant thread keeps a slot clock from those records, derives the busy
 * partitions of the current wall-clock slot (PUSCH of slots w-2..w-4, UL control of w-1, PDSCH of slots w+1, w+2) and
 * enqueues the next segment just in time on the stream of the largest class that fits the free gap, pausing when none
 * does. Configuration by environment (see dapp_tenant.cu); started from cuphycontroller main once the L1 is ready.
 */
#pragma once
#include <string>

namespace nv {
namespace dapp {
namespace tenant {

// Starts the tenant thread when DAPP_TENANT=1 (returns false with err empty when disabled, false with err on failure).
bool start(std::string& err);
void stop();

} // namespace tenant
} // namespace dapp
} // namespace nv
