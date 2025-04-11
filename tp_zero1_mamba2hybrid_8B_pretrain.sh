#!/bin/bash

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --cache_dir=$HOME/neuron_compile_cache/"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

MODEL_SIZE="8B"
# vocab size
VOCAB_SIZE=50288  # change based on tokenizer used to prepare the dataset
# TP degree
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# global batch size
: "${GBS:=1024}"
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=28000    # change as needed
# warmup steps
WARMUP_STEPS=1500   # change as needed
# learning rate
LR=2.0e-4
# data path
DATA_PATH="$1"
# save path
CHECKPOINT_DIR="$2"
# Checkpoint freq
CHECKPOINT_FREQ=5000

#############################################

export NUM_NEURONCORES=32
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    if [ $NODE_ID -eq 0 ]; then
        echo "WORLD_SLURM_NTASKS=$WORLD_SIZE"
        echo "NODE_ID=$NODE_ID"
        echo "MASTER_ADDRESS=$MASTER_ADDRESS"
        echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
    fi
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
fi

echo "WORLD_SLURM_NTASKS=$WORLD_SIZE"
echo "NODE_ID=$NODE_ID"
echo "MASTER_ADDRESS=$MASTER_ADDRESS"

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export NEURON_RT_NUM_CORES=32
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))


if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
elif [ -v PERF_TEST ] && [ $PERF_TEST -gt 0 ]; then
    STEPS_THIS_RUN=100
    OUTPUT_LOG=log_exe-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

echo VOCAB_SIZE=$VOCAB_SIZE
echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo DATA_PATH=$DATA_PATH

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

CMD_PREFIX="torchrun $DISTRIBUTED_ARGS "

SCRIPT_PATH=$SCRIPT_DIR/train.py

CHECKPOINT_TAG=Mamba2Hybrid${MODEL_SIZE}_bf16_zero1_lr${LR}_cossche_seq${SEQUENCE_LEN}_accu${ACC_STEPS}_bs${MBS}_beta09095_wd01

$CMD_PREFIX $SCRIPT_PATH \
    --tp $TP_DEGREE \
    --model Mamba2Hybrid${MODEL_SIZE} \
    --data_dir $DATA_PATH \
    --max_steps $TOTAL_STEPS \
    --dtype bfloat16 \
    --use_zero_1 \
    --grad_accum_usteps $ACC_STEPS \
    --lr $LR \
    --checkpoint_freq $CHECKPOINT_FREQ \
    --checkpoint_dir $CHECKPOINT_DIR \
    --tag $CHECKPOINT_TAG \
    --vocab_size $VOCAB_SIZE \
    --warmup_steps $WARMUP_STEPS \
    --batch $MBS \
    --rmsnorm_across_groups \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --use_zero_1 \
    --sequence_parallel_enabled \
    2>&1 | tee mamba2_${MODEL_SIZE}_trn_train.log
