import sys
import os
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch_xla.core.xla_model as xm
import neuronx_distributed as nxd
import torch.distributed as dist
import argparse
from transformers import GPTNeoXTokenizerFast
from mamba2.modeling_mamba2 import Mamba2ForCausalLM

def test_model_forward(model_path):

    # Load the model and tokenizer
    def get_model():
        model = Mamba2ForCausalLM.from_pretrained(model_path)
        model.eval()
        return model

    tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_path)
    
    # Configure for neuron
    nxd_config = nxd.neuronx_distributed_config()
    model = nxd.initialize_parallel_model(nxd_config, get_model)
    
    # Prepare input
    text = "Hello World!"
    input_ids = tokenizer(text, return_tensors="pt").to(xm.xla_device()).input_ids

    # need to pad the inputs such that seq_len % model.config.conv_kernel is 0
    # also padding strategy should be left (i.e. add pad_token_id to the left of the tensor)
    # moreover the minimum length is model.config.chunk_size
    B, L = input_ids.shape
    pad_len = model.config.chunk_size - (L % model.config.chunk_size)

    input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=tokenizer.pad_token_id)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    # Basic checks
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == 1  # Batch size
    assert outputs.logits.shape[1] == len(input_ids[0])  # Sequence length
    
    print("Forward pass successful!")
    
    if os.environ.get("WORLD_SIZE"):
        xm.rendezvous("test finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model forward pass')
    parser.add_argument('model_path', type=str,
                        help='Path to the model directory')
    args = parser.parse_args()
    os.environ["NEURON_CC_FLAGS"] = " --model-type=transformer -O1"

    if os.environ.get("WORLD_SIZE"):
        dist.init_process_group("xla")
        test_model_forward(args.model_path)
    else:
        print("No process group initialized, cannot run without torchrun")
