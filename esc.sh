#!/bin/bash

dataset=esc
model=EscFusion

teacher=esc-teacher.pt

train() {
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -mf esc-teacher.pt --model ${model} \
        --device $1 \
        -lr 0.001 -ep 50 \
        >logs/esc-teacher.out
}

prune_finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np5-pr$3-lcm$2
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/escnet.yaml -lcm $2 -pr $3 -co \
        -lr 0.0001 -ep 150 -ree 50 -relr 0.0001 \
        >logs/${save_name}.out
}

#train "cuda:0"

prune_finetune "cuda:0" 1 0.85 &
prune_finetune "cuda:1" 1000 0.85
