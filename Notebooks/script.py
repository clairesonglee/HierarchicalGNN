ROOT_PATH = "/data/FNAL/events"

# System imports
import os
import sys

# External imports
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('..')

from Modules.training_utils import model_selector, kaiming_init, load_from_pretrained
from Modules.tracking_utils import eval_metrics

checkpoint_callback = ModelCheckpoint(
    monitor='track_eff',
    mode="max",
    save_top_k=2,
    save_last=True)

def main():
	model_name = input("input model ID/name")
	#model_name = 1
	model = model_selector(model_name)
	kaiming_init(model)

	logger = WandbLogger(project="TrackML_1GeV")
	trainer = Trainer(gpus=1, max_epochs=model.hparams["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model)

def resume():
	training_id = input("input the wandb run ID to resume the run")
	model_path = "{}{}/checkpoints/last.ckpt".format(ROOT_PATH, training_id)
	ckpt = torch.load(model_path)
	model = model_selector(ckpt["hyper_parameters"]["model"], ckpt["hyper_parameters"])
	    
	logger = WandbLogger(project="TrackML_1GeV", id = training_id)
	accumulator = GradientAccumulationScheduler(scheduling={0: 1, 4: 2, 8: 4})
	trainer = Trainer(gpus=1, max_epochs=ckpt["hyper_parameters"]["max_epochs"], gradient_clip_val=0.5, logger=logger, num_sanity_val_steps=2, callbacks=[checkpoint_callback], log_every_n_steps = 50, default_root_dir=ROOT_PATH)
	trainer.fit(model, ckpt_path="{}{}/checkpoints/last.ckpt".format(ROOT_PATH, training_id))

def test():
	inference_config = {
	    "majority_cut": float(input("majority cut (0.5 for loose matching, 0.9 for strict matching, 1.0 for perfect matching")),
	    "score_cut": 0.7
	}
	model_path = "{}{}/checkpoints/".format(ROOT_PATH, input("input the wandb run ID to load model's state dict"))
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

main()
