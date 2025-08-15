from torch.utils.data import Dataset
import torch
import json
import random
# Custom Dataset class
class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            num_samples = 10
            self.data = random.sample(data_list, num_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        instruction_text = item["instruction"]
        label_text = item["output"]
        label = 1 if label_text == 'True' else 0

        encoding = self.tokenizer(
            instruction_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SentencePairPredictDataset(SentencePairDataset):
    def __getitem__(self, idx):
        return self.data[idx]