#!/bin/bash

dataset=cifar100
model=wrn28_10

teacher=cifar100-wrn28_10-teacher.pth.tar

prune_finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np$2-lcm$3-lcp$4
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/wrn28-$2.yaml -lcm $3 -lcp $4 \
        -lr 0.001 -ep 300 -ree 100 -relr 0.0001 \
        >logs/${save_name}.out
}

#prune_finetune "cuda:0" v1 0.01 0 &
#prune_finetune "cuda:1" v1 0.001 0 &
#prune_finetune "cuda:2" v1 0.0001 0
#prune_finetune "cuda:2" v2 0.0001 0 &
#prune_finetune "cuda:3" v2 0.00001 0 ;
#prune_finetune "cuda:2" v3 0.001 0 &
#prune_finetune "cuda:3" v3 0.0001 0 ;

prune_finetune "cuda:2" v1 0.1 0 ;
prune_finetune "cuda:2" v3 0.00001 0 ;