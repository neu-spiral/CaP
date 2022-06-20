#!/bin/bash

dataset=flash
model=InfoFusionThree

teacher=flash-teacher.pt

prune_finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np3-pr$3-maxloss-lcm$2
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/flashnet.yaml -lcm $2 -pr $3 \
        -lr 0.001 -ep 300 -ree 100 -relr 0.001 \
        >logs/test.out
}

prune_finetune "cuda:0" 1 0.67
#prune_finetune "cuda:2" 10 0.67 &
#prune_finetune "cuda:3" 0.1 0.67

#prune_finetune "cuda:0" 0.01 0.67 &
#prune_finetune "cuda:2" 0.001 0.67 &
#prune_finetune "cuda:3" 0.0001 0.67

#prune_finetune "cuda:0" 1 0.75 &
#prune_finetune "cuda:2" 10 0.75 &
#prune_finetune "cuda:3" 0.1 0.75

#prune_finetune "cuda:0" 0.01 0.75 &
#prune_finetune "cuda:2" 0.001 0.75 &
#prune_finetune "cuda:3" 0.0001 0.75
