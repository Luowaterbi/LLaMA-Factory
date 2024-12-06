torchrun \
    --nproc_per_node 8 \
    --node_rank=$deepspeed_node_rank \
    --nnodes=$deepspeed_node_num \
    --rdzv_id=0 \
    --rdzv_endpoint=$deepspeed_master_addr:$deepspeed_master_port \
    src/train.py mine/$1