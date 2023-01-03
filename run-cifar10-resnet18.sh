#!/bin/bash

dataset=cifar10
model=resnet18

teacher=cifar10-resnet18-teacher.pt

prune_finetune() {
    st=$5
    save_name=${dataset}-${model}-$5-np$2-pr$4-lcm$3
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -st ${st} -pfl -lcm $3 -pr $4 -co \
        -lr 0.01 -ep 300 -ree 150 -relr 0.001 \
        >logs/${save_name}.out
}

prune_finetune "cuda:3" v2 0.001 0.75 kernel
