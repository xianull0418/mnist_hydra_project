import lightning.pytorch as pl
import torch.utils.data as data
from torchvision import datasets, transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
            self.train_set, self.val_set = data.random_split(mnist_full, [55000, 5000])
        
        if stage == "test" or stage is None:
            self.test_set = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)