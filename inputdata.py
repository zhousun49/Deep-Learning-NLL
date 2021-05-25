from torch.utils.data import Dataset, DataLoader, SequentialSampler
class InputData(Dataset):
    def __init__(self, token, tag):
        self.token = token
        self.tag = tag

    def __getitem__(self, index):
        return self.token[index], self.tag[index]

    def __len__(self):
        return len(self.token)