#!/bin/bash

dataset=cifar10
model=resnet18

teacher=cifar10-resnet18-teacher.pt

prune_finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np$2-lcm$3-lcp$4
    python -m source.core.run_partition -cfg config/${dataset}-${model}.yaml \
        -lm ${teacher} -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -pfl -lcm $3 -lcp $4 \
        -lr 0.01 -ep 300 -ree 100 -relr 0.0001 \
        >logs/${save_name}.out
}

prune_finetune "cuda:0" v1 0.00001 0 &
prune_finetune "cuda:1" v1 0.00002 0 &
prune_finetune "cuda:2" v1 0.00005 0 ;

prune_finetune "cuda:0" v1 0.0001 0 &
prune_finetune "cuda:1" v1 0.0002 0 &
prune_finetune "cuda:2" v1 0.0005 0 &
prune_finetune "cuda:3" v1 0.001  0 ;

prune_finetune "cuda:0" v1 0.00001 0.0001 &
prune_finetune "cuda:1" v1 0.00002 0.0001 &
prune_finetune "cuda:2" v1 0.00005 0.0001 ;

prune_finetune "cuda:0" v1 0.0001 0.0001 &
prune_finetune "cuda:1" v1 0.0002 0.0001 &
prune_finetune "cuda:2" v1 0.0005 0.0001 &
prune_finetune "cuda:3" v1 0.001  0.0001 ;


prune_finetune "cuda:0" v2 0.00001 0 &
prune_finetune "cuda:1" v2 0.00002 0 &
prune_finetune "cuda:2" v2 0.00005 0 ;

prune_finetune "cuda:0" v2 0.0001 0 &
prune_finetune "cuda:1" v2 0.0002 0 &
prune_finetune "cuda:2" v2 0.0005 0 &
prune_finetune "cuda:3" v2 0.001  0 ;

prune_finetune "cuda:0" v2 0.00001 0.0001 &
prune_finetune "cuda:1" v2 0.00002 0.0001 &
prune_finetune "cuda:2" v2 0.00005 0.0001 ;

prune_finetune "cuda:0" v2 0.0001 0.0001 &
prune_finetune "cuda:1" v2 0.0002 0.0001 &
prune_finetune "cuda:2" v2 0.0005 0.0001 &
prune_finetune "cuda:3" v2 0.001  0.0001 ;





#-kernel-pr_$2
#-lm ${dataset}-${model}-kernel-np$2_hardprune.pt