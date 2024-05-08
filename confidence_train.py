# adapted from Gabriel Corso's DiffDock confidence_train.py script
import gc
import math
import os
import sys
import warnings

import shutil
from datetime import datetime

from argparse import Namespace, ArgumentParser, FileType
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from utils.logging import warn_with_traceback, Logger, lg

from lightning_modules.confidence_module import ConfidenceModule
from models.flowsite_model import FlowSiteModel
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for running on a macbook
import wandb
import torch
from torch_geometric.loader import DataListLoader, DataLoader

from datasets.confidence_dataset import ConfidenceDataset

from utils.parsing import parse_confidence_args



def main_function():
    args = parse_confidence_args()
    assert(args.main_metric_goal == 'max' or args.main_metric_goal == 'min')
    args.run_name_timed = args.run_name + '_' + datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d_%H-%M-%S")
    torch.set_float32_matmul_precision(precision=args.precision)
    os.environ['MODEL_DIR'] = os.path.join('runs', args.run_name_timed)
    os.makedirs(os.environ['MODEL_DIR'], exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stderr)

    # if args.debug:
    # warnings.showwarning = warn_with_traceback

    if args.wandb:
        wandb_logger = WandbLogger(entity='andmcnutt',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args)
    else:
        wandb_logger = None

    # construct loader
    common_args = { 'num_devices': args.num_devices,'inference_steps': args.inference_steps,
                   'samples_per_complex': args.samples_per_complex,
                    'all_atoms': args.all_atoms, 'balance': args.balance,
                   'rmsd_classification_cutoff': args.rmsd_classification_cutoff}
    exception_flag = False
    # try:
    train_dataset = ConfidenceDataset(split_path=args.train_split_path, args=args, **common_args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # except Exception as e:
    #     if 'The generated ligand positions with cache_id do not exist:' in str(e):
    #         print("HAPPENING | Encountered the following exception when loading the confidence train dataset:")
    #         print(str(e))
    #         print("HAPPENING | We are still continuing because we want to try to generate the validation dataset if it has not been created yet:")
    #         exception_flag = True
    #     else: raise e

    val_dataset = ConfidenceDataset(split_path=args.val_split_path, args=args, **common_args)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    lg(f'Train data: {len(train_dataset)}')
    lg(f'Val data: {len(val_dataset)}')

    # # load confidence model and lightning module
    model = FlowSiteModel(args, confidence_mode=True)
    model_module = ConfidenceModule(args=args, model=model, train_data=train_dataset)

    lg(f'Number of Devices:{args.num_devices}')
    trainer = Trainer(logger=wandb_logger,
                        default_root_dir=os.environ['MODEL_DIR'],
                        num_sanity_val_steps=0,
                        log_every_n_steps=args.print_freq,
                        max_epochs=args.n_epochs,
                        enable_checkpointing=True,
                        limit_test_batches=args.limit_test_batches or 1.0,
                        limit_train_batches=args.limit_train_batches or 1.0,
                        limit_val_batches=args.limit_val_batches or 1.0,
                        check_val_every_n_epoch=args.check_val_every_n_epoch,
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[ModelCheckpoint(monitor=f'val/{args.main_metric}', mode=args.main_metric_goal, filename='best', save_top_k=1, save_last=True, auto_insert_metric_name=True, verbose=True)],
                        accelerator="gpu",
                        devices=args.num_devices,
                        strategy='ddp_find_unused_parameters_true'
                      )

    numel = sum([p.numel() for p in model_module.model.parameters()])
    lg(f'Model with {numel} parameters')

    if not args.run_test:
        trainer.fit(model_module, train_loader, val_loader, ckpt_path=args.checkpoint)

    if args.run_test:
        shutil.copy(args.checkpoint, os.path.join(os.environ['MODEL_DIR'], 'best.ckpt'))
    trainer.test(model=model_module, dataloaders=predict_loader, ckpt_path=args.checkpoint if args.run_test else 'best', verbose=True)

if __name__ == '__main__':
    main_function()
