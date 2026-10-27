#!/bin/bash
# n3_decode_block.sh  (night 3, 2026-10-18): JPEG decode position / time / interference measurement while a no-tenant cuPHY Peak run is up.
#   One 300 s Peak R0 run (X=10) through r0_ladder (name N3_DEC); inside it, by wall-clock offsets from the L1-ready stamp:
#     [T+60, T+120)  CPU decode bench, 1/4/8 threads, taskset 64-71 (cv2 / libjpeg-turbo)        -> cpu_bench.txt
#     [T+130, T+190) GPU nvjpeg decode-only load 10 req/s, MPS client 43 %, batch-8 tiles (service path) -> njd_43.txt
#     [T+200, T+260) same at 61 %                                                                   -> njd_61.txt
#   afterwards tenmin_table.py --warm X --window 60 per segment gives B_abs / RU late of each 60 s window vs the quiet windows.
# Needs: cuPHY-CP/dapp_hook/tools/njd_bench (built in the container, see tools/njd_bench.cu) and the venv python for the CPU bench.
set -u
R=/home/troy/build/aerial-cuda-accelerated-ran; S=$R/cuPHY-CP/dapp_hook/sweep/fullstack; O=${FS_OUT:-$R/prof/run66}; cd $R
IMG=$R/cuPHY-CP/dapp_hook/models/img4k; CIMG=/opt/nvidia/cuBB/cuPHY-CP/dapp_hook/models/img4k
cat > $O/ladder_N3.txt <<EOT
DEC pattern=tmp_ddddddsuuu_peak ncells=10 slots=600000 tags=311,223 keep_raw=2
EOT
FS_OUT=$O bash $S/r0_ladder.sh $O/ladder_N3.txt N3 > $O/driver_N3.log 2>&1 &
LAD=$!
for i in $(seq 1 300); do grep -q 'L1 ready' $O/driver_N3.log 2>/dev/null && break; sleep 1; done
T=$(date +%s); echo "L1 ready at $(date +%T) -> decode schedule relative to T=$T" | tee $O/n3_schedule.txt
sleep 60;  echo "cpu bench start $(date +%T)" | tee -a $O/n3_schedule.txt
taskset -c 64-71 ~/.venvs/dapp_yolo/bin/python $R/cuPHY-CP/dapp_hook/tools/jpeg_cpu_bench.py --images $IMG --threads 1,4,8 --rounds 12 > $O/cpu_bench.txt 2>&1
echo "cpu bench end $(date +%T)" | tee -a $O/n3_schedule.txt
while [ $(( $(date +%s) - T )) -lt 130 ]; do sleep 1; done
for cap in 43 61; do
  echo "njd $cap start $(date +%T)" | tee -a $O/n3_schedule.txt
  docker exec -u root -e CUDA_MPS_PIPE_DIRECTORY=/var -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$cap c_aerial_troy bash -lc "taskset -c 64-67 /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/tools/njd_bench /opt/nvidia/cuBB/cuPHY-CP/dapp_hook/e2e/libnvjpeg_dec.so 0 10 60 1280 8 1 $CIMG/uhd_0_q60.jpg $CIMG/uhd_1_q60.jpg $CIMG/uhd_2_q60.jpg $CIMG/uhd_3_q60.jpg" > $O/njd_$cap.txt 2>&1
  echo "njd $cap end $(date +%T): $(tail -1 $O/njd_$cap.txt | cut -c1-200)" | tee -a $O/n3_schedule.txt
  [ $cap = 43 ] && while [ $(( $(date +%s) - T )) -lt 200 ]; do sleep 1; done
done
wait $LAD; echo "ladder done $(date +%T)" | tee -a $O/n3_schedule.txt
