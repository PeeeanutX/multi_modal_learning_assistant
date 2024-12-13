from torch.utils.data import DataLoader, Dataset
from sentence_transformers import InputExample


class CustomDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Example usage
examples = [InputExample(texts=["query", "positive"], label=1.0), InputExample(texts=["query", "negative"], label=0.0)]
dataset = CustomDataset(examples)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    for example in batch:
        print(example)
