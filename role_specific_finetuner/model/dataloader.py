from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch

class DS(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length):
        self.df = pd.read_csv(data_path, sep='\t', )
        self.df = self.df[:len(self.df)]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.languages_map = {
            'en': {"label": "English", 'id': 0},
            'hi': {"label": "Hindi", 'id': 1},
            'te': {"label": "Telugu", 'id': 2}, 
            'bn': {"label": "Bengali", 'id': 3},
            'pa': {"label": "Punjabi", 'id': 4},
            'ur': {"label": "Urdu", 'id': 5}, 
            'or': {"label": "Odia", 'id': 6}, 
            'as': {"label": "Assamese", 'id': 7},
            'gu': {"label": "Gujarati", 'id': 8},
            'mr': {"label": "Marathi", 'id': 9},
            'kn': {"label": "Kannada", 'id': 10},
            'ta': {"label": "Tamil", 'id': 11},
            'ml': {"label": "Malayalam", 'id': 12} 
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_text = self.df.iloc[idx]['input'].replace('S|', '<H>').replace('P|', '<R>').replace('O|', '<T>')
        target_text = self.df.iloc[idx]['target']
        lang_code = self.df.iloc[idx]['lang']
        lang = self.languages_map[lang_code]['label']
        lang_id = self.languages_map[lang_code]['id']
        task = self.df.iloc[idx]['task']

        if task == 'translation':
            prefix = f'translate to {lang} text: '
            # input_encoding = self.role_specific_encoding(prefix, input_text)
            input_encoding = self.tokenizer(prefix+input_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)
            target_encoding = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

            input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
            labels = target_encoding['input_ids']
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

            return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'lang': torch.tensor(lang_id)}   
            # squeeze() is needed to remove the batch dimension
        elif task == 'rdf2text':
            prefix = f'rdf to {lang} text: '
            # input_encoding = self.role_specific_encoding(prefix, input_text)
            input_encoding = self.plain_encoding(prefix, input_text)
            target_encoding = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

            input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
            labels = target_encoding['input_ids']
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

            return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'lang': torch.tensor(lang_id)}   
            # squeeze() is needed to remove the batch dimension
        else:
            print('error in identifying task')

    def role_specific_encoding(self, prefix, input_text):
        input_ids = []
        attention_mask = []
        role_ids = []

        prefix_tokenized = self.tokenizer.encode(prefix)[:-1]   # ignoring the eos token at the end
        input_ids.extend(prefix_tokenized)
        attention_mask.extend([1] * len(prefix_tokenized))
        role_ids.extend([0] * len(prefix_tokenized))
        try:
            # data = json.loads(input_text)
            data = eval(input_text)
        except json.decoder.JSONDecodeError:
            print(input_text)
            raise

        for triple in data:
            subject = triple[0]
            predicate = triple[1]
            object = triple[2]

            subject_tokenized = self.tokenizer.encode(subject)[:-1]
            predicate_tokenized = self.tokenizer.encode(predicate)[:-1]
            object_tokenized = self.tokenizer.encode(object)[:-1]

            input_ids.extend(subject_tokenized)
            attention_mask.extend([1] * len(subject_tokenized))
            role_ids.extend([1] * len(subject_tokenized))

            input_ids.extend(predicate_tokenized)
            attention_mask.extend([1] * len(predicate_tokenized))
            role_ids.extend([2] * len(predicate_tokenized))

            input_ids.extend(object_tokenized)
            attention_mask.extend([1] * len(object_tokenized))
            role_ids.extend([3] * len(object_tokenized))
        
        input_ids.extend([self.tokenizer.eos_token_id])
        input_ids = self.pad_and_truncate(input_ids)
        attention_mask = self.pad_and_truncate(attention_mask)
        role_ids = self.pad_and_truncate(role_ids)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'role_ids': role_ids}
    
    def plain_encoding(self, prefix, input_text):
        try:
            # data = json.loads(input_text)
            data = eval(input_text)
        except json.decoder.JSONDecodeError:
            print(input_text)
            raise
        
        linearized_input = ''
        for triple in data:
            for item in triple:
                linearized_input += str(item)
            
        return self.tokenizer(prefix+linearized_input, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        

    def pad_and_truncate(self, ids):
        if len(ids) > self.max_source_length:
            return torch.tensor(ids[:self.max_source_length])
        else:
            return torch.tensor(ids + [self.tokenizer.pad_token_id] * (self.max_source_length - len(ids)))
            # above line is hacky, i.e. padding with pad_token_id, but it works for now. for attention_mask, it should be 0.

    @staticmethod
    def create_tokenizer(tokenizer_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        tokenizer.add_tokens(['<H>', '<R>', '<T>', '<QH>', '<QR>', '<QT>'])
        print("we added HRT QHRT to the tokenizer")
        return tokenizer

class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = DS.create_tokenizer(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train = DS(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length)
            self.val = DS(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length)
        else:
            self.test = DS(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=0,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=0,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=0,shuffle=False)

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Datamodule args')
        parser.add_argument('--train_path', default='data/train.csv', type=str)
        parser.add_argument('--val_path', default='data/val.csv', type=str)
        parser.add_argument('--test_path', default='data/test.csv', type=str)
        parser.add_argument('--tokenizer_name_or_path', type=str)
        parser.add_argument('--max_source_length', type=int, default=128)
        parser.add_argument('--max_target_length', type=int, default=128)
        parser.add_argument('--train_batch_size', type=int, default=4)
        parser.add_argument('--val_batch_size', type=int, default=4)
        parser.add_argument('--test_batch_size', type=int, default=4)
        return parent_parser