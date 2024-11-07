torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:8000 train.py --data WIKI --group 1 --minibatch_parallelism 1
