import itertools
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial
import copy

from lightning import Trainer
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_scatter import scatter_mean
from datasets.complex_dataset import ComplexDataset
from utils.logging import lg
from utils.train_utils import DuplicateSampler
from lightning_modules.flowsite_module import FlowSiteModule
from models.flowsite_model import FlowSiteModel

class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split):
    cache_path = args.cache_path
    if not args.no_torsion:
        cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'
    split_path = args.split_train if split == 'train' else args.split_val
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                       + ('' if args.no_torsion or args.num_conformers == 1 else
                                           f'_confs{args.num_conformers}')
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings'))
    return cache_path

def get_args(model_config):
    model_args = Namespace(**yaml.full_load(model_config))

    return model_args



class ConfidenceDataset(ComplexDataset):
    def __init__(self, split_path, inference_steps, samples_per_complex, all_atoms,
                 args, balance=False, rmsd_classification_cutoff=2, num_devices=1):

        super(ComplexDataset, self).__init__()

        self.data_source = "pdbbind"
        self.args = args
        self.inference_steps = inference_steps
        self.all_atoms = all_atoms
        self.balance = balance
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.samples_per_complex = samples_per_complex

        #score model info
        self.score_model_ckpt = args.score_model_ckpt
        self.score_model_config = args.score_model_config

        self.confidence_cache_dir_name = f'idxFile{os.path.splitext(os.path.basename(split_path))[0]}--protFile{args.protein_file_name}--ligConRad{args.lig_connection_radius}--confidence_set--{os.path.basename(self.score_model_ckpt).replace(".ckpt","")}'
        self.full_cache_path = os.path.join(args.cache_path, self.confidence_cache_dir_name)
        os.makedirs(self.full_cache_path, exist_ok=True)
        pdb_ids = np.loadtxt(split_path, dtype=str)
        pdb_ids = np.array([f"{pid}_{i}" for pid in pdb_ids for i in range(samples_per_complex)])
        # print(np.random.permutation(len(pdb_ids)))
        # shuffled_pdb_ids = pdb_ids[np.random.permutation(len(pdb_ids))]

        valid_complex_paths_path = os.path.join(self.full_cache_path,'valid_complex_paths.txt')
        lig_meta_data_path = os.path.join(self.full_cache_path,'lig_meta_data.npz')

        valid_paths = []
        lig_sizes = []
        num_contacts = []
        if not os.path.exists(valid_complex_paths_path):
            assert self.score_model_ckpt is not None and self.score_model_config is not None
            valid_paths, lig_sizes, num_contacts = self.preprocessing(split_path, valid_complex_paths_path)
        else:
            for pdb_id in tqdm(pdb_ids, desc="checking for valid complexes"):
                if os.path.exists(os.path.join(self.full_cache_path, pdb_id + ".pt")):
                    valid_paths.append(os.path.join(self.full_cache_path, pdb_id + ".pt"))
                    npz_file = np.load(os.path.join(self.full_cache_path, pdb_id + "lig_meta_data.npz"))
                    lig_sizes.append(npz_file['arr_0'])
                    num_contacts.append(npz_file['arr_1'])

        np.savez(os.path.join(lig_meta_data_path), np.array(lig_sizes,dtype=object), np.array(num_contacts,dtype=object), allow_pickle=True)
        np.savetxt(os.path.join(valid_complex_paths_path), valid_paths, fmt="%s")
        valid_complex_paths = np.loadtxt(valid_complex_paths_path, dtype=str)

        lg('Loading valid complex path names')
        npz_file = np.load(lig_meta_data_path, allow_pickle=True)
        lig_sizes, num_contacts = npz_file['arr_0'], npz_file['arr_1']
        assert len(valid_complex_paths) == len(lig_sizes)
        assert len(valid_complex_paths) > 0
        lg(f'Finished loading combined data of length: {len(valid_complex_paths)}')
        valid_ids = []
        for idx in range(len(valid_complex_paths)):
            if np.any((lig_sizes[idx] <= args.max_lig_size) & (lig_sizes[idx] >= args.min_lig_size)):
                valid_ids.append(idx)
        filtered_paths, filtered_sizes, filtered_contacts = valid_complex_paths[valid_ids], lig_sizes[valid_ids], num_contacts[valid_ids]

        lg(f'Finished filtering combined data for ligands of min size {args.min_lig_size} and max size {args.max_lig_size} to end up with this many: {len(filtered_paths)}')
        self.data_paths = [path for path, size in zip(filtered_paths, filtered_sizes) if any(size <= args.max_lig_size)]
        lg(f'Finished filtering combined data for ligands of max size {args.max_lig_size} to end up with this many: {len(self.data_paths)}')
        self.data_dict = {}

    def preprocessing(self, split_path, valid_complex_paths_path):
        valid_paths = []
        lig_sizes = []
        lig_contacts = []
        # load the data and model to generate the ligand positions and RMSDs
        score_model_args = get_args(self.score_model_config)
        predict_data = ComplexDataset(score_model_args, split_path, data_source=self.data_source,  data_dir="")
        predict_loader = DataLoader(predict_data, batch_size=self.samples_per_complex, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, sampler=DuplicateSampler(predict_data, self.samples_per_complex))
        
        model = FlowSiteModel(score_model_args)
        model_module = FlowSiteModule.load_from_checkpoint(self.score_model_ckpt, args=score_model_args, model=model, map_location="cuda")
        trainer = Trainer(accelerator="gpu", devices=1, strategy='ddp_find_unused_parameters_true')
        predictions = trainer.predict(model=model_module, dataloaders=predict_loader, ckpt_path=self.score_model_ckpt)
        torch.save(predictions,'predictions.pt')
        complex_names = []
        for idx, batch in tqdm(enumerate(predict_loader)):
            _, xt, _, _ = predictions[idx]
            rmsd = self.get_rmsd(batch, batch['ligand'].pos, xt)
            batch['ligand'].pos = xt
            batch['ligand'].rmsd = rmsd
            batch['ligand'].rmsd_class = (rmsd < self.rmsd_classification_cutoff)

            # split the batch into individual complexes
            for i in range(len(batch)):
                complex_data = batch.get_example(i)
                lig_size = np.array(complex_data['ligand'].size)
                lig_sizes.append(lig_size)
                lig_contact = np.array((complex_data['protein'].min_lig_dist < 4).sum(dim=0))
                lig_contacts.append(lig_contact)

                torch.save(complex_data, os.path.join(self.full_cache_path, f'{complex_data["pdb_id"]}_{i}.pt'))
                np.savez(os.path.join(self.full_cache_path, f'{complex_data["pdb_id"]}_{i}lig_meta_data.npz'), lig_size, lig_contact)

                valid_paths.append(os.path.join(self.full_cache_path, f'{complex_data["pdb_id"]}_{i}.pt'))
                complex_names.append(f'{complex_data["pdb_id"]}_{i}')
        return valid_paths, lig_sizes, lig_contacts

    def get_rmsd(self, batch, gt_pos, pred_pos):
        pre_nodes = 0
        num_nodes = batch['ligand'].size
        isomorphisms = batch['ligand'].isomorphisms
        new_idx_x = []
        for i in range(len(batch)):
            cur_num_nodes = num_nodes[i]
            current_isomorphisms = [
                torch.LongTensor(iso).to(pred_pos.device) for iso in isomorphisms[i]
            ]
            if len(current_isomorphisms) == 1:
                new_idx_x.append(current_isomorphisms[0] + pre_nodes)
            else:
                gt_pos_i = gt_pos[pre_nodes : pre_nodes + cur_num_nodes]
                pred_pos_i = pred_pos[pre_nodes : pre_nodes + cur_num_nodes]
                pred_pos_list = []

                for iso in current_isomorphisms:
                    pred_pos_list.append(torch.index_select(pred_pos_i, 0, iso))
                total_iso = len(pred_pos_list)
                gt_pos_i = gt_pos_i.repeat(total_iso, 1)
                pred_pos_i = torch.cat(pred_pos_list, dim=0)
                dist = torch.square(
                    gt_pos_i-pred_pos_i
                )
                # group by isomorphism
                dist = dist.view(total_iso, cur_num_nodes, -1).sum(dim=(1,2))
                min_idx = dist.argmin(dim=0)
                new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)
            pre_nodes += cur_num_nodes
        new_idx = torch.cat(new_idx_x, dim=0)
        pred_pos = pred_pos.index_select(0,new_idx)
        square_devs = torch.square(gt_pos - pred_pos)
        return scatter_mean(square_devs.sum(dim=1), batch["ligand"].batch, -1).sqrt().cpu().numpy()

