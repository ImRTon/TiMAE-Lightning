from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
import json
import lightning as L
import pandas as pd
import numpy as np

class UCR2018(L.LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        normalize: bool,
        dataset_path: str | Path,
        val_dataset_path: str | Path | None = None,
        split_ratio: float = 0.85,
        crop_size: int | None = None,
        pretrain: bool = False,
        std: float | None = None,
        mean: float | None = None,
        num_classes: int | None = None,
        class_weights: list[float] | None = None,
        class_id_start: int | None = None
    ):
        super().__init__()

        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.split_ratio = split_ratio

        self.df = pd.read_csv(dataset_path, header=None, sep='\t')

        if num_classes is None:
            num_classes = len(self.df.iloc[:, 0].value_counts())
        self.num_classes = num_classes

        if class_id_start is None:
            class_id_start = int(self.df.iloc[:, 0].min())
            if class_id_start != 1 and class_id_start != 0:
                raise RuntimeWarning("Class ID should start from 0 or 1, please check the dataset.")
        self.class_id_start = class_id_start

        if class_weights is None:
            class_counts = self.df.iloc[:, 0].value_counts().sort_index()
            class_freq = class_counts / class_counts.sum() * num_classes

            class_weights = 1.0 / class_freq
            class_weights = class_weights / class_weights.sum() * num_classes

            class_weights = [max(0.1, min(w, 10)) for w in class_weights]
        self.class_weights = class_weights

        if normalize:
            if std is None:
                std = float(self.df.values.std(ddof=1))
            if mean is None:
                mean = float(self.df.mean().mean())
            print(f"Using Mean: {mean}, Std: {std}")

        if val_dataset_path is not None:
            if isinstance(val_dataset_path, str):
                val_dataset_path = Path(val_dataset_path)
            self.val_df = pd.read_csv(val_dataset_path, header=None, sep='\t')
        else:
            self.val_df = None

        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == 'fit':
            # Split dataset into train and validation if needed
            
            if self.val_df is None:
                self.train_df, self.val_df = train_test_split(
                    self.df, train_size=self.split_ratio, random_state=42)
            else:
                self.train_df = self.df

            self.train_label = self.train_df.pop(0).to_numpy() - self.class_id_start
            self.val_label = self.val_df.pop(0).to_numpy() - self.class_id_start

            if self.hparams.normalize:
                self.train_df = (self.train_df - self.hparams.mean) / self.hparams.std
                self.val_df = (self.val_df - self.hparams.mean) / self.hparams.std

            self.train_df = self.train_df.to_numpy()
            self.val_df = self.val_df.to_numpy()
            
        elif stage == 'test':
            self.test_label = self.df.pop(0).to_numpy() - self.class_id_start
            self.test_df = self.df.to_numpy()

        elif stage == 'predict':
            self.predict_label = self.df.pop(0).to_numpy() - self.class_id_start
            self.predict_df = self.df.to_numpy()

    def train_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.train_df, 
                self.train_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            num_workers=20
        )
    
    def val_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.val_df, 
                self.val_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=20
        )
    
    def test_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.test_df, 
                self.test_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=20
        )
    
    def predict_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.predict_df, 
                self.predict_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=20
        )
        
class UCR2018Dataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray, is_pretrain: bool, crop_size: int | None = None):
        super().__init__()
        self.data = data
        self.label = label
        self.crop_size = crop_size
        self.is_pretrain = is_pretrain

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.crop_size is not None:
            start = np.random.randint(0, data.shape[0] - self.crop_size)
            data = data[start:start+self.crop_size]
        if self.is_pretrain:
            return torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        else:
            return torch.tensor(data, dtype=torch.float32).unsqueeze(1), torch.tensor(self.label[idx], dtype=torch.int64)
        
class KaggleECGCategorization(L.LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        normalize: bool,
        dataset_path: str | Path,
        val_dataset_path: str | Path | None = None,
        split_ratio: float = 0.85,
        crop_size: int | None = None,
        pretrain: bool = False,
        std: float | None = None,
        mean: float | None = None,
        num_classes: int | None = None,
        class_weights: list[float] | None = None
    ):
        super().__init__()

        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.split_ratio = split_ratio

        self.df = pd.read_csv(dataset_path, header=None)

        if num_classes is None:
            num_classes = len(self.df.iloc[:, -1].value_counts())
        self.num_classes = num_classes

        if class_weights is None:
            class_counts = self.df.iloc[:, -1].value_counts().sort_index()
            class_freq = class_counts / class_counts.sum() * num_classes

            class_weights = 1.0 / class_freq
            class_weights = class_weights / class_weights.sum() * num_classes

            class_weights = [max(0.1, min(w, 10)) for w in class_weights]
        self.class_weights = class_weights

        if normalize:
            if std is None:
                std = float(self.df.values.std(ddof=1))
            if mean is None:
                mean = float(self.df.mean().mean())
            print(f"Using Mean: {mean}, Std: {std}")

        if val_dataset_path is not None:
            if isinstance(val_dataset_path, str):
                val_dataset_path = Path(val_dataset_path)
            self.val_df = pd.read_csv(val_dataset_path, header=None)
        else:
            self.val_df = None

        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == 'fit':
            # Split dataset into train and validation if needed
            
            if self.val_df is None:
                self.train_df, self.val_df = train_test_split(
                    self.df, train_size=self.split_ratio, random_state=42)
            else:
                self.train_df = self.df

            self.train_label = self.train_df.pop(self.train_df.columns[-1]).to_numpy()
            self.val_label = self.val_df.pop(self.val_df.columns[-1]).to_numpy()

            if self.hparams.normalize:
                self.train_df = (self.train_df - self.hparams.mean) / self.hparams.std
                self.val_df = (self.val_df - self.hparams.mean) / self.hparams.std

            self.train_df = self.train_df.to_numpy()
            self.val_df = self.val_df.to_numpy()
            
        elif stage == 'test':
            self.test_label = self.df.pop(self.df.columns[-1]).to_numpy()
            self.test_df = self.df.to_numpy()

        elif stage == 'predict':
            self.predict_label = self.df.pop(self.df.columns[-1]).to_numpy()
            self.predict_df = self.df.to_numpy()

    def train_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.train_df, 
                self.train_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            num_workers=20
        )
    
    def val_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.val_df, 
                self.val_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=20
        )
    
    def test_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.test_df, 
                self.test_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=20
        )
    
    def predict_dataloader(self):
        return DataLoader(
            UCR2018Dataset(
                self.predict_df, 
                self.predict_label, 
                self.hparams.pretrain,
                self.hparams.crop_size
            ), 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            num_workers=20
        )