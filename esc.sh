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
    save_name=${dataset}-${model}-${st}-np$2-pr$3-lcm$4
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/escnet-$2.yaml -pr $3 -lcm $4 -co \
        -lr 0.0001 -ep 150 -ree 50 -relr 0.0001 \
        >logs/${save_name}.out
}

plot() {
    st=kernel
    save_name=test
    lm=${teacher}
    lm=${dataset}-${model}-${st}-np5-pr$3-lcm$4.pt
    lm=${dataset}-${model}-${st}-npv2-pr$3-lcm$4.pt
    
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${lm} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/escnet-$2.yaml -st ${st} -lcm $3 -pr $4 \
        -lr 0.01 -ep 0 -ree 0 -relr 0.001 \
        >logs/test.out
}

#train "cuda:0"

#prune_finetune "cuda:0" v2 0.80 1 &
#prune_finetune "cuda:1" v2 0.80 1000
#prune_finetune "cuda:1" v2 0.85 1000 

plot "cuda:2" v2 0.80 1