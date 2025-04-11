import os
import torch
import torch.distributed as dist
from mamba2.configuration_mamba2 import Mamba2Config
import neuronx_distributed as nxd
import neuronx_distributed.parallel_layers.parallel_state as ps
from mamba2.modeling_mamba2 import Mamba2ForCausalLM
from predefined_configs import get_config
import transformers.modeling_utils as modeling_utils
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import numpy as np

# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


def print_stats(tensor, ref_tensor, tensor_type):
    assert tensor.shape == ref_tensor.shape, f"Tensor {tensor.shape} and ref_tensor {ref_tensor.shape} shapes differ"
    print(f'###################### {tensor_type} ######################')
    print(tensor.dtype, ref_tensor.dtype)

    tensor = tensor.cpu().to(torch.float32)
    ref_tensor = ref_tensor.cpu().to(torch.float32)
    if torch.allclose(tensor, ref_tensor, atol=1e-3, rtol=3e-2):
        print("NKI and Torch MATCH")
    else:
        print("NKI and Torch DIFFER")

    print(f"Max absolute error:  {torch.abs(tensor - ref_tensor).max():.5f}")
    print(f"Mean absolute error: {torch.abs(tensor - ref_tensor).mean():.5f}")
    print(f"norm: {tensor.abs().mean()} ref norm: {ref_tensor.abs().mean()}")
    print(f"Max rel. error:  {(torch.abs(tensor - ref_tensor) / ref_tensor.abs().clamp(min=0.001)).max() * 100:.5f}%")
    print(f"Mean rel. error: {(torch.abs(tensor - ref_tensor) / ref_tensor.abs().clamp(min=0.001)).mean() * 100:.5f}%")

    print("Largest relative discrepancy:")
    idx = (torch.abs(tensor - ref_tensor) / ref_tensor.abs().clamp(min=0.001)).argmax()
    print(
        f"index {np.unravel_index(idx, tensor.shape)} out={tensor.flatten()[idx].item():.5f} ref={ref_tensor.flatten()[idx].item():.5f}")


def run(args, backend):
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = xm.xla_device()

    config = get_config(
        args.model,
        args.vocab_size,
        rmsnorm_within_groups=(not args.rmsnorm_across_groups),
        n_groups=args.n_groups,
        sequence_parallel_enabled=args.sequence_parallel_enabled,
    )

    def get_model():
        model = Mamba2ForCausalLM(config).to(dtype=args.dtype)
        model.eval()
        if args.checkpoint_load:
            state_dict = torch.load(args.checkpoint_path)
            model.load_state_dict(state_dict)
            xm.mark_step()
            print('load sucessfully!')
        # check that weight tying worked
        if model.config.tie_word_embeddings:
            assert model.backbone.embeddings.weight is model.lm_head.weight
        return model

    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=args.tp,
        sequence_parallel=args.sequence_parallel_enabled,
        model_init_config=None,
    )
    model = nxd.initialize_parallel_model(nxd_config, get_model)
    print(f"Mamba2 Model =\n{model}")
    is_root = xm.is_master_ordinal(local=False)
    if is_root:
        print('NEURON_CC_FLAGS: ', os.environ.get('NEURON_CC_FLAGS', None))
        print('XLA_IR_DEBUG: ', os.environ.get('XLA_IR_DEBUG', None))
        print('XLA_HLO_DEBUG: ', os.environ.get('XLA_HLO_DEBUG', None))
        print('TP groups:', ps.get_tensor_model_parallel_group(as_list=True))
        print('DP groups:', ps.get_data_parallel_group(as_list=True))
        print('Arguments: ', args)
        param_size, dtype = 0, None
        for param in set(model.parameters()):
            param_size += param.nelement()
            dtype = param.dtype
        print(f"Model size: {param_size / 10 ** 6:.1f}M parameters/core")
        print(f"Param dtype: {dtype}")


    with torch.no_grad():
        input_ids = torch.randint(0, args.vocab_size, (args.batch, args.seq_len))

        orig_logits = model(input_ids=input_ids.to(model.device)).logits
        xm.mark_step()

        orig_logits = orig_logits.to('cpu')
        if torch.isnan(orig_logits).any():
            print(f"NaN detected in origional logits")
            exit()

        mamba2cache = Mamba2Cache(config, batch_size=args.batch, dtype=torch.bfloat16, device=model.device)
        xm.mark_step()

        logits_list = []
        for i in range(args.seq_len):
            print(i)
            logits = model.step(input_ids[:, i:i + 1].to(model.device), cache_params=mamba2cache)
            xm.mark_step()

            logits = logits.to('cpu')
            xm.mark_step()

            logits_list.append(logits)

            # Check for NaNs
            if torch.isnan(logits).any():
                print(f"NaN detected at step {i}")
                break

            xm.mark_step()
        print('Done')

        stacked_logits = torch.cat(logits_list, dim=1)
        print(f"Stacked logits {stacked_logits.shape}")
        print(f"orig logits {orig_logits.shape}")
        print_stats(stacked_logits, orig_logits, args.dtype)


def init_processes(args, backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(args=args, backend=backend)
    xm.rendezvous("_mp_fn finished")


def tp_loader(state_dict, LOCAL_RANK, tp_size, config):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith('out_proj.weight'):  # row parallel
            dim_1_shape = v.shape[1]
            cv = torch.split(v, dim_1_shape // tp_size, dim=1)
            new_state_dict[k] = cv[LOCAL_RANK]
        elif k.endswith('in_proj_xBC.weight') or 'conv1d' in k:  # xBC and Conv Col para
            wx, wB, wC = torch.split(v, [config.hidden_size * 2, config.n_groups * config.state_size,
                                         config.n_groups * config.state_size],
                                     dim=0)
            wx_tp = torch.split(wx, wx.shape[0] // tp_size, dim=0)[LOCAL_RANK]
            wB_tp = torch.split(wB, wB.shape[0] // tp_size, dim=0)[LOCAL_RANK]
            wC_tp = torch.split(wC, wC.shape[0] // tp_size, dim=0)[LOCAL_RANK]
            xBC_tp = torch.cat((wx_tp, wB_tp, wC_tp), dim=0)
            new_state_dict[k] = xBC_tp
        elif 'norm' in k and 'mixer' not in k:
            new_state_dict[k] = v
        else:  # norm weight and z and dt
            dim_0_shape = v.shape[0]
            rv = torch.split(v, dim_0_shape // tp_size, dim=0)
            new_state_dict[k] = rv[LOCAL_RANK]

    return new_state_dict


if __name__ == '__main__':
    import torch_xla.core.xla_model as xm
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Mamba2Block Configuration")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Data type")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=50272, help="Vocab size")
    parser.add_argument("--model", default="Mamba130M", help="Hugging face model to profile")
    parser.add_argument("--backend", type=str, default="xla", choices=['xla', 'nccl', 'gloo'])
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument("--n_groups", type=int, default=1, help="Number of groups")

    parser.add_argument("--sequence_parallel_enabled", action="store_true", help="Use sequence parallelism.")
    parser.add_argument("--seq_len", type=int, default=2048, help="The sequence length of input.")
    parser.add_argument("--rmsnorm_across_groups", action="store_true",
                        help="Uses (HF style) RMSNorm instead of the custom one that normalizes independently for each of the n_groups.")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable Neuron debugging flags to dump model graph and compiler logs.")
    parser.add_argument("--checkpoint_load", action="store_true", help="Load checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, default="./", help="ckpt loading path")

    os.environ["NEURON_CC_FLAGS"] = " -O1 --auto-cast none --model-type transformer"

    args = parser.parse_args()

    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
    args.dtype = getattr(torch, args.dtype)

    if args.dtype == torch.bfloat16:
        modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16

    if args.debug:
        os.environ["XLA_IR_DEBUG"] = "1"
        os.environ["XLA_HLO_DEBUG"] = "1"
        os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

    init_processes(args, backend=args.backend)
