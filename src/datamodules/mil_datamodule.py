from dataclasses import dataclass
import json
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import dill
import lmdb
import msgpack
import numpy as np
import pandas as pd
from sklearn.model_selection import  StratifiedKFold
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from tqdm.auto import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchvision.transforms import transforms

from src.datamodules.components import BaseDataModule, BaseKFoldDataModule
from src import utils


log = utils.get_pylogger(__name__)


class BaseDataset(Dataset):
    def __len__(self):
        return len(self.keys)

    def __del__(self):
        """Close lmdb."""
        if self.lmdb is not None:
            self.lmdb.close()

    def get_stainings(self, name, stainings):
        if name=="camelyon":
            assert set(stainings).issubset({'HE'}), "Make sure to select a subset of the following staining methods ['HE']"
        return list(set(stainings))
        

        

    def get_label_type(self, name, label_type):
        if name=="camelyon":
            assert label_type in {'cancer_class_short'}, "Make sure to select a subset of the following staining methods ['cancer_class_short']"
        return label_type

    def open_lmdb(self, dataset_dir):
        self.lmdb = lmdb.open(
                dataset_dir,
                map_size=65.535, #16 * 1024 ** 3,
                readonly=True,
                lock=False,
                readahead=False,
                # meminit=False,
                # max_readers=2 ** 16,
            )

    def load_meta_lmdb(self, meta_dir):
        metas = {}
        if os.path.exists(meta_dir):
            with lmdb.open(
                meta_dir,
                map_size=2 ** 16,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2 ** 16) as env:
                with env.begin() as txn:
                    keys = [k.decode("ascii") for k in msgpack.loads(txn.get(b"__keys__"))]
                    for key in keys:
                        value = txn.get(key.encode("ascii"), default = -1)
                        metas.update({
                            f'{key}': dill.loads(value)
                        })
        return metas


class SlideDataset(BaseDataset):
    def __init__(
        self,
        name: str="camelyon",
        root_dir: str = "./data/processed",
        stainings: list = ['HE'],
        level: int = 1,
        feature_extractor: str = "resnet",
        reference_shape: int=1024,
        label_type: str='grade',
        num_classes: int=2,
        shuffle_instances: bool=False,
        load_patch_labels: bool=False
        ) -> None:
        super().__init__()
        self.name = name
        self.level = level
        self.root_dir = f"{root_dir}/{name}"
        self.stainings = self.get_stainings(name, stainings)
        self.metas = self.load_meta_lmdb(meta_dir=self.root_dir + f"/level_{level}/metas")
        self.dataset_dir = self.root_dir + f"/token_lmdb_{feature_extractor}"
        self.keys = self.load_keys()
        self.shape = self.load_shape(reference_shape)
        self.label_type = self.get_label_type(name, label_type)
        self.labels = self.load_labels(self.keys, num_classes)
        if load_patch_labels:
            self.patch_labels = self.load_patch_labels(self.keys)
        self.shuffle_instances = shuffle_instances
        self.lmdb = None

    def __getitem__(self, index):
        # Transfer self.data/patch_pool index to lmdb dataset idx
        if self.lmdb is None:
            self.open_lmdb(self.dataset_dir)
        # transform integer index to file_id
        index = list(self.keys)[index]
        values = []
        for key in self.keys[index]:
            with self.lmdb.begin() as txn:
                token = txn.get(f"{key}".encode("ascii"))
            token = np.frombuffer(token, dtype=np.float32).reshape(self.shape).copy()
            values.append(token)
        if self.shuffle_instances:
            keys_values = list(zip(self.keys[index], values))
            random.shuffle(keys_values)
            self.keys[index], values = zip(*keys_values)
        values = np.stack(values)
        values = torch.Tensor(values)
        label = self.labels[index]
        return values, label, self.keys[index]

    def load_keys(self):
        assert os.path.exists(self.dataset_dir), f"{self.dataset_dir} not found"
        with lmdb.open(
            self.dataset_dir,
            map_size=2 ** 16,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2 ** 16) as env:
            with env.begin() as txn:
                all_keys = sorted([k.decode("ascii") for k in msgpack.loads(txn.get(b"__keys__"))])
        all_keys = [key for key in all_keys for staining in self.stainings if staining in key]
        if os.path.exists(self.dataset_dir+"_keys.json"):
            with open(self.dataset_dir+"_keys.json", "r") as file:
                keys = json.load(file)
        else:
            patient_ids = sorted(self.metas.keys())
            file_ids = sorted([filename.split(".")[0] for i in patient_ids for staining in self.stainings for filename in self.metas[i]['files'].keys() if staining in self.metas[i]['files'][filename]['staining']])
            all_keys_short = [key.split(f"_level")[0] for key in all_keys]
            key_ids = list(sorted(set(all_keys_short)))
            if len(set(key_ids).difference(set(file_ids)))>0:
                keys={}
                all_keys_short = np.array(all_keys_short)
                all_keys_np = np.array(all_keys)
                for key_id in tqdm(key_ids, total=len(key_ids), leave=False, bar_format="{desc:20}{bar:50}{r_bar}{bar:-10b}"):
                    assert len(all_keys_short)==len(all_keys_np)
                    indices = np.where(all_keys_short == key_id)
                    keys.update({f'{key_id}':all_keys_np[indices[0]].tolist()})
                    all_keys_np = np.delete(all_keys_np, indices[0])
                    all_keys_short = np.delete(all_keys_short, indices[0])
            else:
                keys = {f'{file_id}':[key for key in all_keys if file_id in key] for file_id in tqdm(file_ids, total=len(file_ids), leave=False, bar_format="{desc:20}{bar:50}{r_bar}{bar:-10b}")}
            with open(self.dataset_dir+"_keys.json", "w") as outfile:
                json.dump(keys, outfile, indent=4)
        missing_file_ids = []
        for file_id in keys.keys():
            if not keys[file_id]:
                missing_file_ids.append(file_id)
                log.info(f"Missing keys for {file_id}")
        for file_id in missing_file_ids:
            del keys[file_id]
        return keys

    def load_labels(self, file_ids, num_net_classes):
        #global labels
        patient_ids = sorted(self.metas.keys())
        labels = {f'{file_id}':self.metas[i][self.label_type] for i in patient_ids for file_id in file_ids if i in file_id}
        num_dataset_classes = len(set(labels.values()))
        assert num_dataset_classes==num_net_classes, f"Make sure that the number of outputs of the network ({num_net_classes}) match the number of classes in the dataset ({num_dataset_classes})"
        return labels
    
    def load_patch_labels(self, file_ids):
        if os.path.exists(self.dataset_dir+"_patch_labels.json"):
            with open(self.dataset_dir+"_patch_labels.json", "r") as file:
                patch_labels = json.load(file)
        else:
            patch_labels = {}
            key_label_dict = {}
            patient_ids = sorted(self.metas.keys())
            patch_metas = {f'{file_id}':self.metas[i]['files'][file_id+'.tif']['patches'] for i in patient_ids for file_id in file_ids if i in file_id}
            for i in tqdm(patient_ids, total=len(patient_ids), leave=False, bar_format="{desc:20}{bar:50}{r_bar}{bar:-10b}"):
                written_patch_labels = len(patch_labels)
                keys = file_ids[i]
                patches = patch_metas[i]
                label_key = list(patches.keys())[-1]
                assert 'cancer' in label_key and f"level{self.level}" in label_key, f"Did not find matching labels"
                for key in keys: 
                    patch_labels.update({f'{key}': 0})
                if self.labels[i]==1:
                    for x, y, patch_label in tqdm(zip(patches['x_position'], patches['y_position'], patches[label_key]), total=len(patches['x_position']),
                                                leave=False, bar_format="{desc:20}{bar:50}{r_bar}{bar:-10b}"):
                        if patch_label>0.2:
                            key = [key for key in keys if key==f"{i}_HE_level{self.level}_x{x}_y{y}"]
                            if len(key)>0:
                                assert len(key)==1, f"Found multiple matching keys"
                                key = key[0]
                                patch_labels.update({f'{key}': 1})
                assert(len(patch_labels)==len(keys)+written_patch_labels)
            with open(self.dataset_dir+"_patch_labels.json", "w") as outfile:
                    json.dump(patch_labels, outfile, indent=4)
        return patch_labels
    
    def load_shape(self, reference_shape):
        assert os.path.exists(self.dataset_dir), f"{self.dataset_dir} not found"
        with lmdb.open(
            self.dataset_dir,
            map_size=2 ** 16,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=2 ** 16) as env:
            with env.begin() as txn:
                shape = msgpack.loads(txn.get(b"__shape__"))
        assert shape==[reference_shape], log.info(f'Configurations in "datamodule.dataset.reference_shape" are wrong. Change value from {[reference_shape]} to {shape}.')
        return shape

    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dino, dino_v2, swav, name, reference_shape, num_classes, feature_extractor, label_type, root_dir):
        self.dino = dino()
        self.dino_v2 = dino_v2()
        self.swav = swav()
        assert self.dino.keys == self.dino_v2.keys == self.swav.keys, "Keys do not match!"
        self.reference_shape = reference_shape
        self.name = name
        self.num_classe = num_classes
        self.feature_extractor = feature_extractor
        self.label_type = label_type
        self.root_dir = self.dino.root_dir
        self.keys = self.dino.keys
        self.metas = self.dino.metas

    def __getitem__(self, i):
        x_dino, y_dino, keys_dino = self.dino[i]
        x_dino_v2, y_dino_v2, keys_dino_v2 = self.dino_v2[i]
        x_swav, y_swav, keys_swav = self.swav[i]
        values={'dino': x_dino, 'dino_v2': x_dino_v2, 'swav': x_swav}
        assert y_dino == y_dino_v2 == y_swav, "Labels do no match"
        label = y_dino
        assert keys_dino == keys_dino_v2 == keys_swav, "Keys do not match"
        keys = keys_dino
        return values, label, keys

    def __len__(self):
        assert len(self.dino_v2) == len(self.swav)
        return len(self.dino_v2)


class MILDataModule(BaseDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: Dataset = SlideDataset,
        num_splits: int = 5,
        fold_id: int = 0,
        batch_size: int = 64,
        test_size: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super(MILDataModule).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.prepare_data_per_node = False

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # 1. Split dataset into main set and hold-out/test set with equal class distribution
        self.dataset = self.hparams.dataset()
        self.main_test_split_name, self.main_case_ids, self.main_case_labels, test_case_ids = self.get_main_test_split()
        # Define test dataset
        test_indices = [idx for idx, key in enumerate(self.dataset.keys) for test_case_id in test_case_ids if test_case_id in key]
        self.test_dataset = Subset(self.dataset, test_indices)

        self.num_folds = self.hparams.num_splits
        train_val_fold_name = f"{self.num_folds}fold_{self.main_test_split_name}"
        folds_path = self.dataset.root_dir + f"/{train_val_fold_name}"

        
        if os.path.isfile(folds_path):
            with open(folds_path, "r") as file:
                self.folds = json.load(file)
        else:
            self.folds = {}
            if self.num_folds > 1:
                skf = StratifiedKFold(n_splits=self.num_folds, shuffle=False, random_state=None)
                for fold_id, (main_train_idxs, main_val_idxs) in enumerate(skf.split(self.main_case_ids, self.main_case_labels)):
                    self.folds.update({
                        f'{fold_id}': {
                            'train_case_ids':[self.main_case_ids[i] for i in main_train_idxs],
                            'val_case_ids': [self.main_case_ids[i] for i in main_val_idxs]
                        }
                    })
                
            else:
                self.folds.update({
                        f'0': {
                            'train_case_ids': self.main_case_ids,
                            'val_case_ids': test_case_ids
                        }
                    })
            with open(folds_path, "w") as outfile:
                json.dump(self.folds, outfile, indent=4)
        # Extract dataset case ids from fold dictionary
        train_case_ids = self.folds[f'{self.hparams.fold_id}']['train_case_ids']
        val_case_ids = self.folds[f'{self.hparams.fold_id}']['val_case_ids']
        # Get every index in the dataset which correspondes to the case_id. Therefore, compare each dataset key with each case_id
        # Define trainings dataset
        train_indices = [idx for idx, key in enumerate(self.dataset.keys) for train_case_id in train_case_ids if train_case_id in key]
        self.train_fold = Subset(self.dataset, train_indices)
        # Define validation dataset
        val_indices = [idx for idx, key in enumerate(self.dataset.keys) for val_case_id in val_case_ids if val_case_id in key]
        self.val_fold = Subset(self.dataset, val_indices)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_fold,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_fold,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass
