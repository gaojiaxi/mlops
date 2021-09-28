import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer

class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size = 32):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

