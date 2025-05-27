


from tokenizers import Tokenizer
from torch import Tensor
from tqdm import tqdm
import csv
import random

class SentencePairDataset:
    """
    Dataset class for loading Uyghur-English sentence pairs from ./.sentences.csv.
    Each row in the CSV should have Uyghur in the first column and English in the second.
    """
    def __init__(self, tokenizer: Tokenizer, tsv_files: list[str], max_allowed_tokens = 300):
        self.tokenizer = tokenizer
        self.data: list[tuple[list[int], list[int]]] = []
        for tsv in tsv_files:
            with open(tsv, encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t")
                language_pair = next(reader)
                print(f"loading dataset for [{language_pair[0]}]->[{language_pair[1]}] in file: {tsv} ...")
                for row in reader:
                    def append_data(row: tuple[str, str], source_language_code: str, target_language_code: str):
                        if len(row) >= 2:
                            source = f"<SOS><{target_language_code}>{row[0].strip().lower()}</{target_language_code}><EOS>"
                            target = f"<SOS><{target_language_code}>{row[1].strip().lower()}</{target_language_code}><EOS>"
                            source_language: list[int] = tokenizer.encode(source).ids
                            target_language: list[int] = tokenizer.encode(target).ids
                            if len(source_language) > max_allowed_tokens or len(target_language) > max_allowed_tokens:
                                return
                            self.data.append((source_language, target_language))
                            
                            # randomly add same sentence as source and target to maintain language tag consistency.
                            if random.random() > 0.95:
                                self.data.append((target_language, target_language))
                    
                    append_data((row[0], row[1]), language_pair[0], language_pair[1])
                    #the reverse is blocked because single source -> multiple target sentence causes issue.
                    # append_data((row[1], row[0]), language_pair[1], language_pair[0])
                    
        self.data = sorted(self.data, key=lambda item: len(item))
        self.print_data_count()
    
    def print_data_count(self):
        print(f"size of dataset: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def split_into_train_set_validation_set(self, trainset_ratio = 0.95) -> tuple["SentencePairDataset", "SentencePairDataset"]:
        trainset_ratio = min(trainset_ratio, 0.98)
        trainset_ratio = max(trainset_ratio, 0.5)
        train, test = SentencePairDataset(self.tokenizer, []), SentencePairDataset(self.tokenizer, [])
        for i in self.data:
            if random.random() > trainset_ratio:
                test.data.append(i)
            else:
                train.data.append(i)
        
        train.print_data_count()
        test.print_data_count()
        return train, test

def collete_fn(batch: list[tuple[list[int], list[int]]]) -> Tensor:
    """
    first tuple item in batch is Uyghur and second tuple item is English.
    we have to calculate the maximum length of item in batch and make them as tuple of tensor.
    """
    # INSERT_YOUR_CODE
    import torch

    # Separate Uyghur and English sequences
    source_seqs = [item[0] for item in batch]
    target_seqs = [item[1] for item in batch]

    # Find max lengths
    max_source_len = max(len(seq) for seq in source_seqs)
    max_target_len = max(len(seq) for seq in target_seqs)

    # Pad sequences
    def pad_sequence(seqs, max_len, pad_value=0):
        return [
            seq + [pad_value] * (max_len - len(seq))
            for seq in seqs
        ]

    source_padded = pad_sequence(source_seqs, max_source_len)
    target_padded = pad_sequence(target_seqs, max_target_len)

    # Convert to tensors
    source_tensor = torch.tensor(source_padded, dtype=torch.long)
    target_tensor = torch.tensor(target_padded, dtype=torch.long)

    return source_tensor, target_tensor