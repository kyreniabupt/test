from torch.utils.data import Dataset
import torch
import json
import random
# train
class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            self.data = data_list
            #print(len(data_list))
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
    
    
# test
class SentencePairDataset2(Dataset):
    def __init__(self, file_path, tokenizer, max_length, sentence1_str, sentence2_str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.sentence1_str = sentence1_str
        self.sentence2_str = sentence2_str

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                
                if self.sentence2_str == "category_path":
                    item['prompt'] = f"""
                    Determine whether a comma-separated category path matches the user's query intent.
                    The path represents levels of hierarchy from general to specific.  
                    If any level is irrelevant or incorrect, return 0.Otherwise, return 1  .
                    Query:   {item[self.sentence1_str]}
                    Product Categories: {item[self.sentence2_str]}
                    """
                elif self.sentence2_str == "item_title":
                    item['prompt'] = f"""
                    Determine whether a product name matches the user's query intent.
                    The product must completely satisfy the user's search query in all aspects 
                    (including product type, brand, model, attributes, etc.).
                    If any aspect is irrelevant or incorrect, return 0. Otherwise, return 1.
                    Query: {item[self.sentence1_str]}
                    Product Name: {item[self.sentence2_str]}"""
                                    
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        label = item['label']

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(int(label), dtype=torch.long)
        }

class SentencePairPredictDataset2(SentencePairDataset2):
    def __getitem__(self, idx):
        return self.data[idx]