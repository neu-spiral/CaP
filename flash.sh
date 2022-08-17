#!/bin/bash

dataset=flash
model=InfoFusionThree

teacher=flash-teacher.pt

prune_finetune() {
    st=$4
    save_name=${dataset}-${model}-${st}-np3-pr$3-lcm$2
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${teacher} -mf ${save_name}.pt --model ${model} \
        --device $1 -np config/flashnet.yaml -st ${st} -lcm $2 -pr $3 \
        -lr 0.001 -ep 300 -ree 100 -relr 0.001 \
        >logs/${save_name}.out
}

plot() {
    st=$4
    save_name=test
    lm=${teacher}
    lm=${dataset}-${model}-$4-np3-pr$3-lcm$2.pt
    
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -lm ${lm} -mf ${save_name}.pt  \
        --device $1 -np config/flashnet.yaml -st ${st} -lcm $2 -pr $3 \
        -lr 0.01 -ep 0 -ree 0 -relr 0.001 \
        >logs/testt1.out
}

#prune_finetune "cuda:0" 1 0.67 partition
#prune_finetune "cuda:1" 1 0.67 &
#prune_finetune "cuda:2" 10 0.67 &
#prune_finetune "cuda:3" 0.1 0.67

#prune_finetune "cuda:1" 0.01 0.67 &
#prune_finetune "cuda:2" 0.001 0.67 &
#prune_finetune "cuda:3" 0.0001 0.67


#prune_finetune "cuda:0" 1 0.75 &
#prune_finetune "cuda:2" 10 0.75 &
#prune_finetune "cuda:3" 0.1 0.75

#prune_finetune "cuda:0" 0.01 0.75 &
#prune_finetune "cuda:2" 0.001 0.75 &
#prune_finetune "cuda:3" 0.0001 0.75

plot "cuda:3" 10 0.67 kernel
plot "cuda:3" 0.1 0.67 kernel
plot "cuda:3" 0.001 0.67 kernel

plot "cuda:3" 10 0.75 kernel
plot "cuda:3" 0.1 0.75 kernel
plot "cuda:3" 0.001 0.75 kernel