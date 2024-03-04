ROOT_PATH = "/data/FNAL/events/"

# System imports
import os
import sys

sys.path.append('..')
import pyarrow as pa
from Modules.training_utils import model_selector, kaiming_init, load_from_pretrained
from Modules.tracking_utils import eval_metrics

# External imports
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler


checkpoint_callback = ModelCheckpoint(
    monitor='track_eff',
    mode="max",
    save_top_k=2,
    save_last=True)

def main():
	#model_name = input("input model ID/name")
	model_name = "4"
	model = model_selector(model_name)
	kaiming_init(model)

	#logger = WandbLogger(project="TrackML_1GeV")
	logger = None
	trainer = Trainer(gpus=1, max_epochs=model.hparams["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model)

def resume():
	#training_id = input("input the wandb run ID to resume the run")
	training_id = "1pxhnaa6"
	model_path = "{}{}{}/checkpoints/last.ckpt".format(ROOT_PATH, "TrackML_1GeV/", training_id)
	print('Model path = ', model_path)
	ckpt = torch.load(model_path)
	print('Checkpoint model name = ', ckpt["hyper_parameters"]["model"])
	model = model_selector(ckpt["hyper_parameters"]["model"], ckpt["hyper_parameters"])
	#model = model_selector("4")
	    
	#logger = WandbLogger(project="TrackML_1GeV", id = training_id)
	logger = None
	accumulator = GradientAccumulationScheduler(scheduling={0: 1, 4: 2, 8: 4})
	trainer = Trainer(gpus=1, max_epochs=ckpt["hyper_parameters"]["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model, ckpt_path=model_path)
'''
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for param_tensor in own_state:
          print(param_tensor, "\t", own_state[param_tensor].size())
        layers = ['node_encoder','edge_encoder','output_layer']
        layer_num = 2
        layer_types = ['weight','bias']
        for layer in layers:
          for i in range(layer_num):
            for layer_type in layer_types:
              param_name = layer+'.'+str(i)+'.'+layer_type
              print('Param name = ', param_name)
              test = own_state[param_name]
              #del own_state['param_name']
              print('test pop = ', test)

        for name, param in state_dict.items():
          if name not in own_state:
            continue
          if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            own_state[name].copy_(param)
'''
def switch():
	# Load input and setup logger
	training_id = "TrackML_1GeV/1pxhnaa6"
	logger = WandbLogger(project="TrackML_1GeV")
	#logger = None

	# Load checkpoint from Hierarchical Pooling NN
	model_path = "{}{}/checkpoints/last.ckpt".format(ROOT_PATH, training_id)
	ckpt = torch.load(model_path)
	print('Model path = ', model_path)
	print('Checkpoint model name = ', ckpt["hyper_parameters"]["model"])

	# Initialize model and parameters
	model_name = "4"
	model = model_selector(model_name)
 	#model = model_selector("4")
	kaiming_init(model)
	
	# Load pretrained parameters from checkpoint when possible

	#print('Initialized model param length = ', len(model.state_dict()))
	#print('Checkpointed model param length = ', len(prev_state_dict))

	num_init_params = 11
	prev_state_dict = ckpt["state_dict"]
	curr_state_dict = model.state_dict()
	plen, clen = len(prev_state_dict)-num_init_params, len(curr_state_dict)
	for i in range(1, plen):
	  print('Curr idx = ', -i)
	  prev_param = list(prev_state_dict)[-i]
	  curr_param = list(curr_state_dict)[-i]
	  print('Compare param names = ', prev_param, curr_param)
	  assert prev_param == curr_param
	  param = prev_state_dict[prev_param].data
	  curr_state_dict[curr_param].copy_(param) 

	# Setup model for training
	model.load_state_dict(curr_state_dict, strict=False)
	#optimizer.load_state_dict(chkpt['optimizer_state_dict'])
	#loss = chkpt['loss']
	#accumulator = GradientAccumulationScheduler(scheduling={0: 1, 4: 2, 8: 4})
	trainer = Trainer(gpus=1, max_epochs=ckpt["hyper_parameters"]["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model)

#----------------------------------------------------------------------------------------
def test():
	inference_config = {
	    "majority_cut": 1.0,
	    #"majority_cut": float(input("majority cut (0.5 for loose matching, 0.9 for strict matching, 1.0 for perfect matching")),
	    "score_cut": 0.7
	}
	training_id = "TrackML_1GeV/1pxhnaa6"
	model_path = "{}{}/checkpoints/".format(ROOT_PATH, training_id)
	model_paths = os.listdir(model_path)
	model_paths.remove("last.ckpt")
	ckpt_name = model_paths[0]
	for i in model_paths:
	  if int(i.strip("epoch=").split("-")[0]) > int(ckpt_name.strip("epoch=").split("-")[0]):
	    ckpt_name = i
	model_path = os.path.join(model_path, ckpt_name)

	ckpt = torch.load(model_path)
	sweep_configs = {**(ckpt["hyper_parameters"]), **inference_config}

	model = model_selector(ckpt["hyper_parameters"]["model"], sweep_configs)
	    
	model = load_from_pretrained(model, ckpt = ckpt)
	model.setup("test")
	trainer = Trainer(gpus=1)
	test_results = trainer.test(model, model.test_dataloader())[0]
'''
main()
print("Training new model")
'''
#resume()
#test()
switch()
print("Resuming training on model")
#'''
