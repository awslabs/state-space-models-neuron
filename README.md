## State Space Models for AWS Neuron

This repository implements the necessary code to train a [Mamba-2 model](https://arxiv.org/abs/2405.21060) on AWS Trainium instances using the
`neuronx-distributed` library. The model implemented is compatible with
HuggingFace's [Mamba-2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)
and with the [weights](https://huggingface.co/state-spaces/mamba2-2.7b) from `state-spaces/mamba2`
(see below for the conversion script).

The core SSM kernels have been rewritten and optimized for AWS Trainium using the
[Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) library.

The training code uses the `neuronx-distributed` training library (see also
the [LLaMA 8B training example](https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama/tp_zero1_llama_hf_pretrain/tp_zero1_llama_hf_pretrain.py)).
It supports tensor parallelism, Zero-1 optimizer, gradient accumulation, mixed precision,
and multi-instance training. We refer to
the [neuronx-distributed documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html)
for additional usage information and tutorials.

## Setup
**Packages:** The code has been tested using the `aws_neuronx_venv_pytorch_1_13` venv on 	
Deep Learning AMI Neuron (Ubuntu 22.04), which uses `torch-neuronx==1.13.1` and `neuronx-cc-2.15`.
See `requirements.txt` for additional dependencies.

**Dataset:** Run the following script to download and tokenize the `wikicorpus` dataset for training. Set `num_proc` as
appropriate.
**Note:** `seq_len` must be divisible by `chunk_size` of the model (128 by default)
```sh
seq_len=512
python tokenize_dataset.py \
  --seq_len ${seq_len} --save_path ./data/wikicorpus_${seq_len} --dataset "wikicorpus" \
  --dataset_config "raw_en" --tokenizer "EleutherAI/gpt-neox-20b" --num_proc ${num_proc}
```

## Run a training job

To start a simple dry run with tensor parallelism, `bfloat16` precision and `zero1` optimizer use the following command
(you will need to adjust the hyperparameters for an actual training run).
**Note:** The sequence length is automatically inferred from the dataset.
```sh
torchrun --nproc_per_node 32 train.py  --tp 8  --model Mamba7B  --data_dir ./data/wikicorpus_512/  --max_steps 16  --dtype bfloat16  --use_zero_1  --grad_accum_usteps 1  --lr 0.001  --checkpoint_freq 8  --checkpoint_dir ./model_save/  --tag mamba2_7b_testrun  --vocab_size 50288  --warmup_steps 5  --batch 1  --rmsnorm_across_groups  --beta1 0.9  --beta2 0.95  --weight_decay 0.1
```
The following script shows additional configuration options and can be adapted for multi-instance training.
```sh
bash tp_zero1_mamba2_7B_pretrain.sh DATA_DIR MODEL_SAVE_DIR
```

## For Developers

The model implementation is in the folder `mamba2`. This contains:

**Kernels**

* `mamba2_kernel.py` implements the forward and backward Mamba2 kernel for Trainium using NKI.
* `conv1d_grouped.py` is a custom kernel to replace the standard `nn.Conv1d`. It is optimized for grouped convolutions
  with small kernel size.

**Model**

* `mamba2_mixer.py` implements the main Mamba2 block. It is based
  on [the official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py#L191)
  but replaces the CUDA kernels with NKI kernels, and replaces the linear layers with `neuronx-distrubuted`'s
  `ParallelLinear` layers to enable tensor parallelism. It also implements a different `Mamba2RMSNormGated` that
  normalizes each group independently (like in the official Mamba-2 repo, but unlike the `transformer`'s version). This
  is
  needed to ensure consistency when using tensor parallelism, but we leave a config option to keep the HuggingFace behavior
  if needed (e.g., to load existing checkpoints with `n_groups>1`).
* `modeling_mamba2.py` implements the HuggingFace wrapper. The main changes
  are the use of `ParallelEmbedding` and `RowParallelLinear` for the embedding and final layer, again to support tensor
  parallelism.

## Compatibility Notes

**Compatibility with HuggingFace and `state-spaces` models.** The `Mamba2ForCausalLM` wrapper in this repo is compatible with the
[one in `transformers`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)
except that it splits the `in_proj` linear layer into 3 separate layers.

The script `convert_mamba2_ssm_checkpoint.py` can be used
to convert the PyTorch Mamba2 checkpoint from `state-spaces/mamba2` into the right HuggingFace format for our repo.
This will save the converted checkpoints in a new local folder.

```sh
model_name="mamba2-130m"
checkpoint_path="./checkpoints_xla"
original_checkpoint_path="./checkpoints_original"
mkdir -p "${checkpoint_path}"
huggingface-cli download "state-spaces/${model_name}" --local-dir "${original_checkpoint_path}/${model_name}"
torchrun --nproc_per_node=1 convert_mamba2_ssm_checkpoint.py -m mamba_ssm -p fp16 -i "${original_checkpoint_path}/${model_name}" -o "${checkpoint_path}/${model_name}" --split_proj
```

You can later load the checkpoint as usual:

```python
Mamba2ForCausalLM.from_pretrained('./checkpoints_xla/mamba2-130m')
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

