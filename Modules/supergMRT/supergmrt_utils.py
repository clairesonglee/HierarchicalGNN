import sys
import yaml
import math
sys.path.append("../../Modules/")


def process_hparams(hparams):    
    if hparams["hidden"] == "ratio":
        hparams["hidden"] = hparams["hidden_ratio"]*hparams["latent"]
    
    if "cluster_granularity" not in hparams:
        hparams["cluster_granularity"] = 0
    
    return hparams

def kaiming_init(model):
    for name, param in model.named_parameters():
        try:
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif name.endswith("0.weight"):  # The first layer does not have ReLU applied on its input
                param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
            else:
                param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))
        except IndexError as E:
            continue

def transfer_params(prev_state_dict):
    with open(path + "BipartiteClassification/Configs/HGNN_GMM.yaml") as f:
      hparams = yaml.load(f, Loader=yaml.FullLoader)
    sweep_configs = {}
    model = BC_HierarchicalGNN_GMM(process_hparams({**hparams, **sweep_configs}))
    kaiming_init(model)

    curr_state_dict = model.state_dict()
    num_init_params = 11
    plen, clen = len(prev_state_dict)-num_init_params, len(curr_state_dict)
    for i in range(1, plen):
      prev_param = list(prev_state_dict)[-i]
      curr_param = list(curr_state_dict)[-i]
      assert prev_param == curr_param
      param = prev_state_dict[prev_param].data
      curr_state_dict[curr_param].copy_(param)




