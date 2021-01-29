import argparse
from typing import Optional
from utils.load_checkpoint import load_checkpoint
from utils.macs_calculation import calculate_TT_macs,calculate_linear_macs
from utils.print_utils import print_net

def get_module_prefix_suffix(s):
    split_point = s.rfind('.') if 'parameters' not in s else s.rfind('.parameters')
    return s[:split_point],s[split_point+1:]

def module_filter(ckpt):
    net_info = {}
    for key in ckpt:
        if 'version' in ckpt:
            continue
        module_name,module_part_suffix = get_module_prefix_suffix(key)
        if module_name not in net_info:
            net_info[module_name] = {}
        net_info[module_name][module_part_suffix] = ckpt[key]
    #print(net_info['decoder.layers.5.fc1'])
    return net_info
    
def ensemble_net_info(net_info):
    consumption = {}
    # 2modes, TT and Linear
    full_macs = 0
    full_params = 0
    for key,sub_info in net_info.items():
        cores = []
        param_sum = 0
        macs_sum = 0
        for module_name,param in sub_info.items():
            if 'bias' in module_name:
                param_sum += param.numel()
                macs_sum += param.numel()//2

            elif 'parameters' in module_name and len(param.shape) == 4:
                cores.append(param)
                param_sum += param.numel()
            
            elif 'weight' in module_name:
                param_sum += param.numel()
                macs_sum += calculate_linear_macs(param,seq_len=1)
        if len(cores) >0:
            macs_sum += calculate_TT_macs(cores,seq_len=1)
        #print(key,"  ",macs_sum)
        full_macs+= macs_sum
        full_params += param_sum

        if macs_sum >0:
            consumption[key] = {'name' : key, 'macs': macs_sum, 'params':param_sum}
    
    consumption['aggregated_results'] = {'macs':full_macs,'params':full_params}
    return consumption

           

def main(args):
    ckpt = load_checkpoint(args.path,args.key)
    cores = []
    net_info = module_filter(ckpt)
    consumption = ensemble_net_info(net_info)
    print_net(consumption)
    # for k in ckpt:
    #     print(k)
    #     if 'fc1.parameters' in k and 'layers.0' in k and 'encoder' in k:
    #         print(k)
    #         print(ckpt[k].shape)
    #         cores.append(ckpt[k])
    # print(calculate_TT_macs(cores,seq_len=1))

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="path of checkpoint file", type=str)
parser.add_argument('--key',help="key to the dictionary of checkpoint", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)