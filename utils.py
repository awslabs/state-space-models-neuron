import math
import queue
import time
import sys
import os
import inspect
import requests
from collections import namedtuple
from dataclasses import astuple
import torch
from transformers import PreTrainedModel

from torch.utils.tensorboard import SummaryWriter
from predefined_configs import CONFIGS_KWARGS

Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])

def dummy_inputs(model: PreTrainedModel, seq_len: int, batch_size: int = 1, device=None):
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    return {
        'input_ids': input_ids,
        'labels': input_ids
    }
    
    
def get_mixed_precision_config(use_zero_1):
    if use_zero_1:
        return {
            "use_master_weights": True,
            "use_fp32_grad_acc": True,
            "use_master_weights_in_ckpt": False,
        }
    return {}
    

class Logger:
    def __init__(self, args, model_dtype):
        xla = "torch_xla" in sys.modules
        world_size = torch.distributed.get_world_size()
        self.throughputs = []
        self.tb = SummaryWriter(
            os.path.join(
                args.output_dir,
                f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                f"_{args.model}"
                f"_{model_dtype}"
                f"_w{world_size}"
                f"_lr{args.lr}"
                f"_bs{args.batch}"
                f"_acc{args.grad_accum_usteps}"
                f"_max{args.max_steps}"
                f"_xla{xla}"
                f"_{self.get_instance_type()}",
            )
        )
        self.tb.add_text("script", "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0)

    def get_instance_type(self):
        try:
            token = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            )
            data = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token": token.text},
            )
            return data.text
        except Exception:
            return os.environ.get("HOSTNAME", "unknown")
        
    
    def log(self, step, step_loss, learning_rate, step_time=None, throughput=None, MFU=None):
        time_now = time.asctime()
        step_time_msg = f"step time : {step_time:.4f} s" if step_time else ""
        throughput_msg = f"throughput : {throughput:.4f} seq/s" if throughput else ""
        MFU_msg = f"MFU : {MFU:.4f}" if MFU else ""
        
        print(
            f"LOG {time_now} - step: {step}, step_loss : {step_loss:.8f} "
            f"learning_rate : {learning_rate:.2e} "
            f"{step_time_msg} ",
            f"{throughput_msg} ",
            f"{MFU_msg} ",
            flush=True,
        )
        self.tb.add_scalar("step loss", step_loss, step)
        self.tb.add_scalar("learning rate", learning_rate, step)
        if step_time:
            self.tb.add_scalar("step time", step_time, step)
        if throughput:
            self.tb.add_scalar("throughput", throughput, step)
        if MFU:
            self.tb.add_scalar("MFU", MFU, step)


class Throughput:
    def __init__(
        self, 
        batch_size, 
        world_size, 
        grad_accum_usteps, 
        tp_size = None, 
        model = None,
        seq_length=2048,
        chip_flops=95, 
        moving_avg_window_size=10, 
        logging_interval=1,
        MFU=True,
        MFU_backpropagation=True,
    ):
        """
        Used to calculate the throughput over a moving window. It records the step time
        between two calls and uses that time to calculate the throughput.
        """
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps * logging_interval
        self.moving_avg_window_size = math.ceil(moving_avg_window_size / logging_interval)
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()
        
        # MFU calculation
        self.batch_size = batch_size * grad_accum_usteps
        self.seq_length = seq_length
        self.tp_size = tp_size
        self.chip_flops = chip_flops
        self.MFU = MFU
        if self.MFU:
            self.MFU_backpropagation = MFU_backpropagation
            self.MFU_factor = self.set_mamba_MFU_factor(model)
            
    def set_mamba_MFU_factor(self, model):
        assert model in CONFIGS_KWARGS
        D, layers, N = astuple(CONFIGS_KWARGS[model])
        L = self.seq_length
        
        # https://quip-amazon.com/w5B1AwLU0wsI/Mamba-MFU-Profiling
        per_layer_flops = 12 * L * D**2 + 8 * L * D + 2 * L * N * D
        total_flops = per_layer_flops * layers * self.batch_size / 10 ** 12
        unit_system_flops = self.chip_flops * self.tp_size
        MFU_factor = total_flops / unit_system_flops
        if self.MFU_backpropagation:
            MFU_factor *= 3
        return MFU_factor

    def _get_MFU(self, step_time):
        return self.MFU_factor / step_time

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        avg_window_time = self.window_time / window_size
        throughput = self.seqs_per_iteration / avg_window_time
        MFU = self._get_MFU(avg_window_time) if self.MFU else None
        return avg_window_time, throughput, MFU
