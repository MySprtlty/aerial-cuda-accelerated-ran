#!/bin/bash

# Check if cuBB_SDK is defined, if not, use default path
cuBB_Path="${cuBB_SDK:-/opt/nvidia/cuBB}"

#./insmod.sh
cd "$cuBB_Path" || exit 1

# Restart MPS
# Export variables
export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_MPS_PIPE_DIRECTORY=/var
export CUDA_MPS_LOG_DIRECTORY=/var

# Stop existing MPS
echo quit | sudo -E nvidia-cuda-mps-control

# Start MPS
sudo -E nvidia-cuda-mps-control -d
sudo -E echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control

configFile=${cuBB_SDK}/cuPHY-CP/cuphycontroller/config/config_benetal.yaml
ls /opt/nvidia/cuBB/cuPHY-CP/cuphycontroller/config/
#Change this to the MAC address of the ORU
#sudo -E sed -i "s/ dst_mac_addr:.*/ dst_mac_addr: 8c:1f:64:d1:15:11/" ${configFile}
# Uncomment for below config. Because aerial is mounted as a volume the $configFile needs to be reverted.
#config="UL-Heavy" # Commenting this line won't revert the change
if [[ $config = "UL-Heavy" ]]; then
	echo "Doing UL-heavy"
	sudo -E sed -i "s/shm_log_level: 4/shm_log_level: 5/" ${configFile}
	sudo -E sed -i "s/pusch_aggr_per_ctx:.*/pusch_aggr_per_ctx: 12/" ${configFile}
	sudo -E sed -i "s/prach_aggr_per_ctx.*/prach_aggr_per_ctx: 4/" ${configFile}
	sudo -E sed -i "s/ul_input_buffer_per_cell:.*/ul_input_buffer_per_cell: 20/" ${configFile}
fi

export AERIAL_LOG_PATH=/var/log/aerial
sudo -E "$cuBB_Path"/build.$(uname -m)/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf config_benetal
sudo -E ./build.$(uname -m)/cuPHY-CP/gt_common_libs/nvIPC/tests/pcap/pcap_collect nvipc /tmp
sudo -E mv /tmp/nvipc*.pcap $AERIAL_LOG_PATH

#Uncomment this if using multiple nvipc interfaces
#sudo ./build/cuPHY-CP/gt_common_libs/nvIPC/tests/pcap/pcap_collect nvipc1 /tmp
#sudo mv /tmp/nvipc*.pcap /var/log/aerial/
#sleep infinity
