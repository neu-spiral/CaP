#!/bin/bash

dataset=cifar10
model=resnet18

teacher=cifar10-resnet18-teacher.pt

prune_finetune() {
    pr=$2
    st=partition
    save_name=${dataset}-${model}-pr_$2
    python -m source.core.run_partition -cfg config/${dataset}-${model}.yaml \
        --device $1 -lm ${teacher} -mf ${save_name}.pt \
        -pr ${pr} -st $st -lt weight \
        -lr 0.01 -ep 300 -ree 100 -relr 0.001 \
        >logs/${save_name}.out
}

pruneMask() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np$2
    python -m source.core.run_partition -cfg config/${dataset}-${model}.yaml \
        -lm ${teacher} -mf ${save_name}.pt  \
        --device $1 -np $2 -lc 0.0001 \
        -lr 0.01 -ep 300 -ree 100 -relr 0.0001 \
        >logs/${save_name}.out
}

pruneMask "cuda:0" 2 &
pruneMask "cuda:1" 4 &
pruneMask "cuda:2" 8 &
pruneMask "cuda:3" 16
#prune_finetune "cuda:0" 0.5 
#-kernel-pr_$2