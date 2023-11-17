from abc import ABC, abstractmethod
import json
import os

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class BaseDataModule(LightningDataModule):
    def get_main_test_split(self):
        # If file exist load ids and labels from disc
        main_test_split_name = self.get_main_test_split_name()
        main_test_split_path = self.dataset.root_dir + f"/{main_test_split_name}"
        if os.path.isfile(main_test_split_path):
            with open(main_test_split_path, "r") as file:
                main_test_split = json.load(file)
            main_case_ids = main_test_split['main']
            main_case_labels = np.array([self.dataset.metas[i][self.dataset.label_type] for i in main_case_ids])
            test_case_ids = main_test_split['test']
        else:
            # Split dataset into main set and hold-out/test set with equal class distribution
            main_test_split = {}
            case_ids = np.array(list(self.dataset.metas.keys()))
            labels = np.array([self.dataset.metas[i][self.dataset.label_type] for i in case_ids])
            main_case_ids, main_case_labels, test_case_ids = self.custom_main_test_split(self.dataset.metas, case_ids, labels)
            main_case_ids = main_case_ids.tolist()
            test_case_ids = test_case_ids.tolist()
            # Save split as json
            main_test_split.update({
                'test': test_case_ids,
                'main': main_case_ids,
            })
            if 'camelyon' in self.dataset.name:
                main_test_split_name = f"main_{len(main_case_ids)}_test_{len(test_case_ids)}_split.json"
                with open(self.dataset.root_dir+f"/{main_test_split_name}", "w") as outfile:
                    json.dump(main_test_split, outfile, indent=4)
            else:
                with open(main_test_split_path, "w") as outfile:
                    json.dump(main_test_split, outfile, indent=4)
        return main_test_split_name, main_case_ids, main_case_labels, test_case_ids 

    def get_main_test_split_name(self):
        # Define name for json-file to save main and hold-out/test split 
        if 'camelyon' in self.dataset.name:
            main_test_split_name = f"main_270_test_129_split.json"
        else:
            assert self.hparams.test_size < 1 and self.hparams.test_size > 0
            test_size_label = int(100*self.hparams.test_size)
            main_size_label = 100 - test_size_label
            main_test_split_name = f"main_{main_size_label}_test_{test_size_label}_split.json"
        return main_test_split_name
    
    def custom_main_test_split(self, metas, case_ids, labels):
        if "camelyon" in self.dataset.name:
            test_case_ids = np.array([case_id for case_id in case_ids if "test" in case_id])
            main_case_ids = np.array(list(set(case_ids).difference(set(test_case_ids))))
            main_case_labels = np.array([metas[i][self.dataset.label_type] for i in main_case_ids])
        else:
            main_case_ids, test_case_ids, main_case_labels, _ = train_test_split(case_ids, labels, test_size=self.hparams.test_size, stratify=labels)
        return main_case_ids, main_case_labels, test_case_ids

class BaseKFoldDataModule(BaseDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass