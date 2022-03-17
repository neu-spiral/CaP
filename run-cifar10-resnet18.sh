#!/bin/bash

dataset=cifar10
model=resnet18

teacher=cifar10_resnet18_teacher.pt

prune_finetune() {
    pr=$2
    st=partition
    save_name=${dataset}-${model}-test
    python -m source.core.run_partition -cfg config/${dataset}-${model}.yaml \
        --device $1 -mf ${save_name}.pt \
        -pr ${pr} -st $st -lt regular \
        -lr 0.1 --lr-scheduler cosine -ep 300 -reopt adam -ree 100 -relr 0.001 \
        >logs/${save_name}.out
}

prune_finetune "cuda:0" 0.5
#prune_finetune "cuda:1" 0.25 &
#prune_finetune "cuda:2" 0.125
#-lm ${teacher} 