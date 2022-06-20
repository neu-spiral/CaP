#!/bin/bash

dataset=cifar10
model=resnet18

teacher=cifar10-resnet18-teacher.pt

train() {
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -mf cifar10-resnet18-teacher.pt --model ${model} \
        --device $1 \
        -lr 0.01 -ep 300 \
        >logs/cifar10-resnet18-teacher.out
}

prune_finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np$2-pr$4-lcm$3
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -pfl -lcm $3 -pr $4 -co \
        -lr 0.01 -ep 300 -ree 100 -relr 0.0001 \
        >logs/${save_name}.out
}

#train "cuda:2"

prune_finetune "cuda:0" v2 0.000001 0.75 &
prune_finetune "cuda:1" v2 0.00001 0.75 &
prune_finetune "cuda:2" v2 0.0001 0.75 &
prune_finetune "cuda:3" v2 0.001 0.75





#-kernel-pr_$2
#-lm ${dataset}-${model}-kernel-np$2_hardprune.pt