#!/bin/bash
torchrun_cmd="torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:8000 train.py --data WIKI --group 2"
$torchrun_cmd &
pids=$!
echo $pids > torchrun_pids.txt
stop_torchrun() {
    if [ -f torchrun_pids.txt ]; then
        pids=$(cat torchrun_pids.txt)
        for pid in $pids; do
            kill -15 $pid
            sleep 2
            if ps -p $pid > /dev/null; then
                kill -9 $pid
            fi
        done
        rm torchrun_pids.txt
    fi
}
trap stop_torchrun SIGINT
wait
