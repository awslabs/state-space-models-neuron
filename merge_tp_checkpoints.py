import torch
import os
import argparse
import json

def merge_tp_ckpts(merge_list, config):
    tp_size = len(merge_list)
    state_dict = merge_list[0]
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k.endswith('out_proj.weight'): # row parallel
            cv = torch.cat([merge_list[LOCAL_RANK][k] for LOCAL_RANK in range(tp_size)], dim=1)
            new_state_dict[k] = cv
        elif 'conv1d' in k:  # Conv Col para
            x_merge = []
            B_merge = []
            C_merge = []
            for LOCAL_RANK in range(tp_size):
                v = merge_list[LOCAL_RANK][k]
                wx, wB, wC = torch.split(v, 
                                         [
                                             config.hidden_size*2 // tp_size, 
                                             config.state_size, 
                                             config.state_size
                                         ],
                                         dim=0)
                x_merge.append(wx)
                B_merge.append(wB)
                C_merge.append(wC)
            wx_tp = torch.cat(x_merge, dim=0)
            wB_tp = torch.cat(B_merge, dim=0)
            wC_tp = torch.cat(C_merge, dim=0)
            xBC_tp = torch.cat([wx_tp, wB_tp, wC_tp], dim=0)
            new_state_dict[k] = xBC_tp
        elif 'in_proj' in k:
            imme_merge = []
            x_merge = []
            B_merge = []
            C_merge = []
            head_merge = []
            for LOCAL_RANK in range(tp_size):
                v = merge_list[LOCAL_RANK][k]
                wi, wx, wB, wC, wh = torch.split(v, 
                                         [
                                             config.hidden_size*2 // tp_size,
                                             config.hidden_size*2 // tp_size, 
                                             config.state_size, 
                                             config.state_size,
                                             config.num_head // tp_size
                                         ],
                                         dim=0)
                imme_merge.append(wi)
                x_merge.append(wx)
                B_merge.append(wB)
                C_merge.append(wC)
                head_merge.append(wh)
            wi_tp = torch.cat(imme_merge, dim=0)
            wx_tp = torch.cat(x_merge, dim=0)
            wB_tp = torch.cat(B_merge, dim=0)
            wC_tp = torch.cat(C_merge, dim=0)
            w_head_merge_tp = torch.cat(head_merge, dim=0)

            tp = torch.cat([wi_tp, wx_tp, wB_tp, wC_tp, w_head_merge_tp], dim=0)
            new_state_dict[k] = tp         
        elif 'norm' in k and 'mixer' not in k:
            new_state_dict[k] = v
        else: # norm weight and z and dt,  Col para
            rv = torch.cat([merge_list[LOCAL_RANK][k] for LOCAL_RANK in range(tp_size)], dim=0)
            new_state_dict[k] = rv
    
    return new_state_dict


class SevenBiConfig:
    def __init__(self):
        self.hidden_size = 4096
        self.state_size = 128
        self.num_head =  (self.hidden_size * 2) // self.state_size
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mamba2_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the tp ckpts (e.g. `dp_rank_00_tp_rank_0x.pt`) to be merged.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    sevenb_config = SevenBiConfig()
    merge_list = [torch.load(f'{args.mamba2_checkpoint_directory}/dp_rank_00_tp_rank_0{m}_pp_rank_00.pt') for m in range(8)]
    result = merge_tp_ckpts(merge_list, sevenb_config)
    torch.save(result, f'{args.output_dir}/pytorch_model.bin')
