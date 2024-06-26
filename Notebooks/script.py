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
    save_top_k=-1,
    save_last=True)

def main():
	#model_name = input("input model ID/name")
	model_name = "5"
	model = model_selector(model_name)
	kaiming_init(model)

	logger = WandbLogger(project="TrackML_1GeV")
	#logger = None
	#trainer = Trainer(gpus=1, max_epochs=model.hparams["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	#trainer = Trainer(gpus=1, max_epochs=model.hparams["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH, limit_train_batches=1)
	trainer = Trainer(gpus=1, max_epochs=100, gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model)

def resume():
	#training_id = input("input the wandb run ID to resume the run")
	training_id = "1pxhnaa6"
	model_path = "{}{}{}/checkpoints/last.ckpt".format(ROOT_PATH, "TrackML_1GeV/", training_id)
	ckpt = torch.load(model_path)
	model = model_selector(ckpt["hyper_parameters"]["model"], ckpt["hyper_parameters"])
	#model = model_selector("4")
	    
	#logger = WandbLogger(project="TrackML_1GeV", id = training_id)
	logger = None
	accumulator = GradientAccumulationScheduler(scheduling={0: 1, 4: 2, 8: 4})
	trainer = Trainer(gpus=1, max_epochs=ckpt["hyper_parameters"]["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model, ckpt_path=model_path)

#----------------------------------------------------------------------------------------
def update(save_ckpt):
	# Load input and setup logger
	#training_id = "TrackML_1GeV/1pxhnaa6"
	#training_id = "TrackML_1GeV/90y92jp7"
	training_id = "TrackML_1GeV/208iaq7z"
	#logger = WandbLogger(project="TrackML_1GeV")
	logger = None

	# Load checkpoint from Hierarchical Pooling NN
	#model_path = "{}{}/checkpoints/last.ckpt".format(ROOT_PATH, training_id)
	model_path = "{}{}/checkpoints/epoch=49-step=15000.ckpt".format(ROOT_PATH, training_id)
	ckpt = torch.load(model_path)

	# Initialize model and parameters
	model_name = "4"
	model = model_selector(model_name)
 	#model = model_selector("4")
	kaiming_init(model)

	#print("Checkpoint keys = ", ckpt.keys())
	#print("Checkpoint hp = ", ckpt["callbacks"])

	# Load pretrained parameters to new state dictionary
	num_init_params = 11
	prev_state_dict = ckpt["state_dict"]
	curr_state_dict = model.state_dict()
	plen, clen = len(prev_state_dict)-num_init_params, len(curr_state_dict)
	for i in range(1, plen):
	  prev_param = list(prev_state_dict)[-i]
	  curr_param = list(curr_state_dict)[-i]
	  assert prev_param == curr_param
	  param = prev_state_dict[prev_param].data
	  curr_state_dict[curr_param].copy_(param) 

	# Load pretrained optimizer to new optimizer
	'''
	prev_opt = ckpt["optimizer_states"][0]
	opt_state = prev_opt['state']
	opt_param_groups = prev_opt['param_groups']
	print((type(opt_state)))
	print((type(opt_param_groups)))

	param_ids = opt_state.keys()
	param_vals = opt_state.values()
	'''
	#params = opt_param_groups['params']
	#print(len(param_vals))
	#for item in param_vals:
	#  print(item.keys(), len(item.values()))
	#print(param_ids)
	#for item in opt_param_groups:
	#  print(item)
	'''
	print('=========================================================')
	curr_opt = torch.optim.AdamW(
                model.parameters(),
                lr=(0.001),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
	print(type(curr_opt))
	checkpoint = {"optimizer_states": curr_opt}
	torch.save(checkpoint, 'checkpoint.pth')
	ckpt = torch.load('checkpoint.pth')
	curr_opt = ckpt["optimizer_states"]
	print(curr_opt)
	opt_state = curr_opt['state']
	opt_param_groups = curr_opt['param_groups']
	print((type(opt_state)))
	print((type(opt_param_groups)))

	param_ids = opt_state.keys()
	param_vals = opt_state.values()
	print(len(param_vals))
	for item in param_vals:
	  print(item.keys(), len(item.values()))
	print(param_ids)
	for item in opt_param_groups:
	  print(item)
	'''
	#curr_opt.load_state_dict(prev_opt.state_dict())
	# Save updated params to new checkpoint file 
	if save_ckpt:
	  ckpt["state_dict"] = curr_state_dict
	  model_path = "{}{}/checkpoints/updated.ckpt".format(ROOT_PATH, training_id)
	  print('Saving checkpoint to path: ', model_path)
	  torch.save(ckpt, model_path)
	curr_opt = None
	return curr_state_dict, curr_opt

def switch(state_dict, optimizer, save_ckpt):
	# Load input and setup logger
	training_id = "TrackML_1GeV/208iaq7z"
	#training_id = "TrackML_1GeV/90y92jp7"
	logger = WandbLogger(project="TrackML_1GeV")
	#logger = None
	if save_ckpt:
	  model_path = "{}{}/checkpoints/updated.ckpt".format(ROOT_PATH, training_id)
	else:
	  #model_path = "{}{}/checkpoints/last.ckpt".format(ROOT_PATH, training_id)
	  model_path = "{}{}/checkpoints/epoch=49-step=15000.ckpt".format(ROOT_PATH, training_id)
	ckpt = torch.load(model_path)

	# Initialize model and parameters
	model_name = "4"
	model = model_selector(model_name)

	# Setup model for training
	if not save_ckpt:
	  model.load_state_dict(state_dict, strict=False)
	  #optimizer.load_state_dict(optimizer)
	  #optimizer, scheduler = trainer.configure_optimizers()
	  #loss = chkpt['loss']
	accumulator = GradientAccumulationScheduler(scheduling={0: 1, 4: 2, 8: 4})
	#trainer = Trainer(gpus=1, max_epochs=ckpt["hyper_parameters"]["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer = Trainer(gpus=1, max_epochs=300, gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	if save_ckpt:
	  trainer.fit(model, ckpt_path=model_path)
	else:
	  trainer.fit(model)

#----------------------------------------------------------------------------------------
def test():
	inference_config = {
	    "majority_cut": 1.0,
	    #"majority_cut": float(input("majority cut (0.5 for loose matching, 0.9 for strict matching, 1.0 for perfect matching")),
	    "score_cut": 0.7
	}
	#training_id = "TrackML_1GeV/1pxhnaa6"
	training_id = "TrackML_1GeV/90y92jp7"
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
#resume()
#test()
'''
save_ckpt = False
state_dict, optimizer = update(save_ckpt)
switch(state_dict, optimizer, save_ckpt)
#'''
