#!/bin/bash

dataset=cifar100
model=wrn28_10

teacher=cifar100-wrn28_10-teacher.pth.tar

finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np$2-lcm$3-lcp$4-v1-finetune
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${dataset}-${model}-kernel-np$2-lcm$3-lcp$4-v1_hardprune.pt -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/wrn28-$2.yaml -lcm $3 -lcp $4 \
        -lr 0.001 -ep 0 -ree 100 -relr 0.00001 \
        >logs/${save_name}.out
}

prune_finetune() {
    st=kernel
    save_name=test
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/wrn28-$2.yaml -lcm $3 -lcp $4 \
        -lr 0.001 -ep 1 -ree 1 -relr 0.00001 \
        >logs/${save_name}.out
}



prune_finetune "cuda:0" v1 0.00001 0 ;