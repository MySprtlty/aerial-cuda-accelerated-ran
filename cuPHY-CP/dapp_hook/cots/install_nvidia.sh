#!/bin/bash
# install_nvidia.sh [target]  - copy this package to the nvidia account (default /home/nvidia/dgist_cots) and hand it over (owner nvidia, scripts executable).
# The package is self-contained on the host side (yamls, OAI compose + gnb.conf, core scripts); the L1 itself runs via 'docker exec' inside the Aerial
# container, so the nvidia account needs only the docker group (it has it) and sudo for the core network scripts.
set -eu; D=$(cd "$(dirname "$0")" && pwd); T=${1:-/home/nvidia/dgist_cots}
sudo mkdir -p $T; sudo cp -a $D/. $T/; sudo rm -rf $T/runs; sudo mkdir -p $T/runs; sudo rm -f $T/runs/current_run
sudo chown -R nvidia:nvidia $T; sudo chmod 755 $T $T/*.sh $T/core/*.sh; sudo chmod 775 $T/runs
echo "installed in $T (owner nvidia): $(ls $T | tr '\n' ' ')"
