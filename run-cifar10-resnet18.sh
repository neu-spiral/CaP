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

finetune() {
    st=kernel
    save_name=${dataset}-${model}-kernel-np$2-pr$5-lcm$3-lcp$4
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${save_name}_hardprune.pt -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -pfl -lcm $3 -lcp $4 -pr $5 -co \
        -lr 0.01 -ep 0 -ree 100 -relr 0.001 \
        >logs/${save_name}_finetune.out
}

prune_finetune() {
    st=$5
    save_name=${dataset}-${model}-$5-np$2-pr$4-lcm$3-pmax
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -st ${st} -pfl -lcm $3 -pr $4 -co \
        -lr 0.01 -ep 300 -ree 150 -relr 0.001 \
        >logs/${save_name}.out
}

prune_finetune_woco() {
    st=$5
    save_name=${dataset}-${model}-$5-np$2-pr$4-lcm$3-woco
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -st ${st} -pfl -lcm $3 -pr $4 \
        -lr 0.01 -ep 150 -ree 150 -relr 0.001 \
        >logs/${save_name}.out
}

plot() {
    st=$5
    save_name=test
    lm=${dataset}-${model}-$5-np$2-pr$4-lcm$3-woco.pt
    lm=${teacher}
    lm=${dataset}-${model}-$5-np$2-pr$4-lcm$3.pt
    lm=${dataset}-${model}-$5-np$2-pr$4-lcm$3-pmax.pt
    
    
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${lm} -mf ${save_name}.pt  \
        --device $1 -np config/resnet18-$2.yaml -st ${st} -pfl -lcm $3 -pr $4 \
        -lr 0.01 -ep 0 -ree 0 -relr 0.001 \
        >logs/test1.out
}


#train "cuda:2"

#prune_finetune "cuda:1" v2 0 0.75 partition &
#prune_finetune "cuda:2" v2 0.000001 0.75 partition &
#prune_finetune_woco "cuda:3" v2 0.000001 0.75 partition

#prune_finetune_woco "cuda:1" v2 0.0000001 0.75 kernel &
#prune_finetune_woco "cuda:2" v2 0.000001 0.75 kernel &
#prune_finetune_woco "cuda:3" v2 0.00001 0.75 kernel
#prune_finetune_woco "cuda:1" v2 0.0001 0.75 kernel &
#prune_finetune_woco "cuda:2" v2 0.001 0.75 kernel &
#prune_finetune_woco "cuda:3" v2 0.01 0.75 kernel

#prune_finetune_woco "cuda:1" v2 0.005 0.75 kernel &
#prune_finetune "cuda:2" v2 0.000005 0.75 kernel &
#prune_finetune "cuda:3" v2 0.0001 0.75 kernel

#plot "cuda:3" v2 0 0.75 partition
#plot "cuda:2" v2 0 0.75 partition
#plot "cuda:2" v2 0.000001 0.7 kernel
#plot "cuda:2" v2 0.000001 0.8 kernel
#plot "cuda:2" v2 0.000001 0.85 kernel
#plot "cuda:2" v2 0.0000001 0.75 kernel
#plot "cuda:2" v2 0.000001 0.75 kernel
#plot "cuda:2" v2 0.00001 0.75 kernel
#plot "cuda:2" v2 0.0001 0.75 kernel
#plot "cuda:3" v2 0.00001 0.75 kernel

#plot "cuda:2" v2 0.00001 0.75 kernel
#plot "cuda:2" v2 0.0001 0.75 kernel
#plot "cuda:2" v2 0.001 0.75 kernel
#plot "cuda:2" v2 0.01 0.75 kernel


#prune_finetune "cuda:3" v2 0.00001 0.75 kernel
prune_finetune "cuda:3" v2 0.001 0.75 kernel

#-kernel-pr_$2
#-lm ${dataset}-${model}-kernel-np$2_hardprune.pt