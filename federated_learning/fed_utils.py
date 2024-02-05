import torch
import copy 

def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(fed_args, global_dict):

    if fed_args.fed_alg in ['scaffold']:
        global_auxiliary = {}               # c in SCAFFOLD
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key])
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # c_i in SCAFFOLD
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]    # delta c_i in SCAFFOLD

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*fed_args.num_clients
        auxiliary_delta_dict = [None]*fed_args.num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict