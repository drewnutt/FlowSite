import csv
import time

import torch, wandb, os, copy
import numpy as np
import pandas as pd
from lightning_modules.general_module import GeneralModule

from collections import defaultdict
from datetime import datetime

from utils.logging import lg

def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log

class ConfidenceModule(GeneralModule):
    def __init__(self, args, model, train_data=None):
        super().__init__(args, model)
        os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
        self.args = args
        self.train_data = train_data

        self.model = model

        self._log = defaultdict(list)
        self.inference_counter = 0

    def on_save_checkpoint(self, checkpoint):
        checkpoint['fake_ratio_scheduler'] = self.fake_ratio_scheduler
        checkpoint['fake_ratio_storage'] = self.fake_ratio_storage
    def on_load_checkpoint(self, checkpoint):
        if 'fake_ratio_scheduler' in checkpoint:
            self.fake_ratio_scheduler = checkpoint['fake_ratio_scheduler']
            self.fake_ratio_storage = checkpoint['fake_ratio_storage']
            if self.train_data is not None:
                self.train_data.fake_lig_ratio = self.fake_ratio_storage.param_groups[0]['lr']

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        out = self.general_step_oom_wrapper(batch, batch_idx)
        self.print_iter_log()
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.stage = "val"
        out = self.general_step_oom_wrapper(batch, batch_idx)
        self.print_iter_log()
        return out

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.stage = "pred"
        out = self.general_step_oom_wrapper(batch, batch_idx)
        self.print_iter_log()
        return out

    def print_iter_log(self):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            print('Run name:', self.args.run_name)
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = gather_log(log, self.trainer.world_size)
            if self.trainer.is_global_zero:
                lg(str(self.get_log_mean(log)))
                if self.args.wandb:
                    wandb.log(self.get_log_mean(log))
            self.log_dict(self.get_log_mean(log), batch_size=1, sync_dist=bool(self.args.num_devices > 1)) #
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    def lg(self, key, data):
        log = self._log
        log["iter/" + key].extend(data)
        log[self.stage + "/" + key].extend(data)

    def general_step(self, batch, batch_idx):
        if self.args.use_true_pos: assert self.args.residue_loss_weight == 1, 'You probably do not want the true positions as input if you are predicting them'
        logs = {}
        batch.logs = logs
        start_time = time.time()
        if self.args.debug:
            lg(f'PDBIDs: {batch.pdb_id}')
            lg(f'Id of ligands lig_choice_id: {batch["ligand"].lig_choice_id}')
            lg(f'Lig name: {batch["ligand"].name}')
            lg(f'fake_lig_id: {batch["protein"].fake_lig_id}')
        if batch['ligand'].size.sum() == len(batch['ligand'].size):
            lg(f'All ligands had size 1 in the batch. Skipping the batch with {batch.pdb_id} with ligand {batch["ligand"].name}')
            return None

        # forward pass
        try:
            pred = self.model(batch)
        except Exception as e:
            lg("Error forward pass")
            lg(batch.pdb_id)
            raise e

        if self.args.rmsd_prediction:
            labels = torch.cat([graph.rmsd for graph in batch]).to(device) if isinstance(batch, list) else batch.rmsd
            confidence_loss = F.mse_loss(pred, labels)
        else:
            if isinstance(self.args.rmsd_classification_cutoff, list):
                labels = torch.cat([graph.rmsd_class for graph in batch]).to(device) if isinstance(batch, list) else batch.rmsd_class
                confidence_loss = F.cross_entropy(pred, labels)
            else:
                labels = torch.cat([graph.rmsd_class for graph in batch]).to(device) if isinstance(batch, list) else batch.rmsd_class
                confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)


        with torch.no_grad():
            self.lg("loss", confidence_loss.cpu().numpy())
            self.lg("symmetric_rmsd", batch.rmsd.cpu().numpy())
            self.lg("conf_score", pred.cpu().numpy())

        return confidence_loss.mean()

    def get_log_mean(self, log):
        out = {}
        out['trainer/global_step'] = float(self.trainer.global_step)
        out['epoch'] = float(self.trainer.current_epoch)

        temporary_log = {}
        aggregated_log = {}
        if self.stage == "pred" and not 'iter_name' in list(log.keys()):
            for key, value in log.items():
                if isinstance(value, list) and len(value) == len(log['pred/num_res']):
                    aggregated_list = []
                    for i in range(max(log['pred/batch_idx']) + 1):
                        values_for_batch = np.array(value)[np.where(np.array(log['pred/batch_idx']) == i)[0]]
                        aggregated_list.append(values_for_batch.reshape(self.args.num_inference, -1))
                    temporary_log[key] = np.concatenate(aggregated_list, axis=1)
                else:
                    aggregated_log[key] = value
        else:
            aggregated_log = log

        for key in aggregated_log:
            try:
                out[key] = np.mean(aggregated_log[key])
            except:
                pass
        return out

    def on_train_epoch_end(self):
        self.on_epoch_end("train")

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_epoch_end("pred")

    def on_epoch_end(self, stage):
        log = self._log
        log = {key: log[key] for key in log if f"{stage}/" in key}
        log = gather_log(log, self.trainer.world_size)
        mse, auroc, pearson_r, kendalltau = get_confidence_metrics(log[f'{stage}/symmetric_rmsd'], log[f'{stage}/conf_score'], self.args.score_output)
        log[f'{stage}/confidence_mse'] = mse
        log[f'{stage}/confidence_auroc'] = auroc
        log[f'{stage}/confidence_pearson_r'] = pearson_r
        log[f'{stage}/confidence_kendalltau'] = kendalltau
        del log[f'{stage}/symmetric_rmsd']
        del log[f'{stage}/conf_score']
        del self._log[f'{stage}/symmetric_rmsd']
        del self._log[f'{stage}/conf_score']
        log['invalid_grads_per_epoch'] = self.num_invalid_gradients
        self.log_dict(self.get_log_mean(log), batch_size=1, sync_dist=bool(self.args.num_devices > 1)) #
        if self.trainer.is_global_zero:
            print('Run name:', self.args.run_name)
            lg(str(self.get_log_mean(log)))
            if self.args.wandb:
                wandb.log(self.get_log_mean(log), step=self.trainer.global_step)
            path = os.path.join(os.environ["MODEL_DIR"], f"{stage}_{self.trainer.current_epoch}.csv")
            log_clone = copy.deepcopy(log)
            for key in list(log_clone.keys()):
                if f"{stage}/" in key or 'invalid_grads_per_epoch' in key or 'confidence' in key:
                    del log_clone[key]
            pd.DataFrame(log_clone).to_csv(path)
        for key in list(log.keys()):
            if f"{stage}/" in key and 'confidence' not in key:
                del self._log[key]

        self.num_invalid_gradients = 0


