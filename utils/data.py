import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import polars as pl
import math

DATA_PATH = os.path.join(os.getcwd(),"data")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class RandomLinearProjectionMNIST(Dataset):
    def __init__(
        self,
        orig_mnist_dataset,
        num_tasks=10,  # Number of augmented versions to create
        seq_len=100,
        labels_shifted_by_one=False,
        spare_mem=False,
        is_train=True,
        data_path="dataset.parquet"
    ):
        self.orig_mnist_dataset = orig_mnist_dataset
        self.data_path = data_path
        self.ext = "train" if is_train else "test"
        self.num_tasks = num_tasks
        self.labels_shifted_by_one = labels_shifted_by_one
        self.spare_mem = spare_mem
        self.seq_len = seq_len
        self.length = len(orig_mnist_dataset) + num_tasks
        self.batch_size = 60000
        self.create_augmented_tasks()
        self.curr_index = 0
        self.dataset = pl.read_parquet(f"dataset_{self.curr_index}_{self.ext}.parquet")

    def create_augmented_tasks(self):
        # Original data is considered task 0
        original_images = [self.orig_mnist_dataset[i][0].view(784) for i in range(len(self.orig_mnist_dataset))]
        original_images = [(transformed_image - transformed_image.mean()) / (transformed_image.std() + 1e-16) for transformed_image in original_images]
        original_labels = [self.orig_mnist_dataset[i][1] for i in range(len(self.orig_mnist_dataset))]
        self.create_and_save_df(original_images, original_labels, 0)

        original_images = []
        original_labels = []
        num_batches = math.ceil(self.num_tasks / self.batch_size)
        num_workers = min(4, num_batches)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Creating a future for each task
            futures = [executor.submit(self.perform_batch_op, batch_idx) for batch_idx in range(num_batches)]
        return
    
    def perform_batch_op(self, batch):
        start_index = batch * self.batch_size
        original_images = []
        original_labels = []
        end_index = min((batch + 1) * self.batch_size, self.num_tasks)
        futures = [self.process_task(task_idx) for task_idx in range(start_index, end_index)]
        for i, future in enumerate(futures):
            transformed_image, permuted_label = future
            original_images.append(transformed_image)
            original_labels.append(permuted_label.item())
        self.create_and_save_df(original_images, original_labels, batch+1)

    def get_dataframe(self, idx):
        if idx == 0:
            file_index = 0
        else:
            file_index = math.ceil(idx * self.seq_len / self.batch_size)
        if (file_index >= self.curr_index):
            self.dataset = pl.read_parquet(f"dataset_{file_index}_{self.ext}.parquet")
            self.curr_index = file_index
        return self.dataset

    def create_and_save_df(self,images, labels, index):
        df = pl.DataFrame({
            "images": [image.numpy() for image in images],
            "labels": labels
        })
        df = df.with_columns(pl.all().shuffle(seed=1))
        filename = f"dataset_{index}_{self.ext}.parquet"
        df.write_parquet(filename)
        print(f"Wrote file {filename}")
        del df
    
    def process_task(self, task_idx):
        rand_idx = np.random.randint(0, len(self.orig_mnist_dataset))
        lin_transform = self.generate_lin_transform(rand_idx)
        label_perm = self.generate_label_perm(rand_idx)
        
        image, label = self.orig_mnist_dataset[rand_idx]
        transformed_image = lin_transform @ image.view(784)
        transformed_image = (transformed_image - transformed_image.mean()) / (transformed_image.std() + 1e-16)
        permuted_label = label_perm[label]

        return transformed_image, permuted_label
    
    def generate_lin_transform(self, task_idx):
        generator = torch.Generator().manual_seed(task_idx)
        return torch.normal(0, 1/784, (784, 784), generator=generator)
    
    def generate_label_perm(self, task_idx):
        generator = torch.Generator().manual_seed(task_idx)
        return torch.randperm(10, generator=generator)
    
    def __len__(self):
        return math.floor(self.length / self.seq_len)
    
    def __getitem__(self, idx):
        # Efficiently fetch batch from disk
        df = self.get_dataframe(idx)
        batch_df = df.slice(idx * self.seq_len, self.seq_len)
        
        x = torch.tensor(batch_df['images'], dtype=torch.float32)
        y = torch.tensor(batch_df['labels'], dtype=torch.int64)
        try:
            if self.labels_shifted_by_one:
                # append labels to images ((x1,0), (x2,y1), ..., (xn-1, yn-2), (xn, yn-1)) - all except the first one
                y_shifted = torch.cat((torch.zeros(size=(1, 10)), F.one_hot(y[:-1], num_classes=10)), dim=0) # (seq_len, 10)
                x = torch.cat((x, y_shifted), dim=1) # (seq_len, 784 + 10), y (seq_len,)
            else:
                # append labels to images ((x1,y1), (x2,y2), ..., (xn-1, yn-1), (xn, 0)) - all except the last one
                y_masked_last = torch.cat((F.one_hot(y[:-1], num_classes=10), torch.zeros(size=(1, 10))), dim=0) # (seq_len, 10)
                x = torch.cat((x, y_masked_last), dim=1)
                y = y[-1] # (10,)
        except Exception as e:
            print(e)
        return x, y
    
    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])



def select_from_classes(x, y, classes_to_select):
    samples_mask = np.array([s in classes_to_select for s in y])
    return x[samples_mask,:], y[samples_mask]


# Data transformations and loading - MNIST
def get_mnist_data_loaders(batch_size=32, flatten=False, drop_last=True, only_classes=None, img_size=28):
    DATA_PATH = os.path.join(os.getcwd(),"data")
    # build transforms
    img_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if img_size < 28 and img_size >= 24:
        img_transformation.transforms.append(transforms.Resize(img_size))
    elif img_size < 24:
        img_transformation.transforms.append(transforms.CenterCrop(24))
        img_transformation.transforms.append(transforms.Resize(img_size))
    if flatten:
        img_transformation.transforms.append(transforms.Lambda(lambda x: torch.flatten(x)))

    train_dataset = datasets.MNIST(DATA_PATH, train=True, download=False, transform=img_transformation)
    if only_classes != None: # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(train_dataset.targets, only_classes if type(only_classes) == torch.Tensor else torch.tensor(only_classes))
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    
    test_dataset = datasets.MNIST(DATA_PATH, train=False, download=False, transform=img_transformation)
    if only_classes != None: # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(test_dataset.targets, only_classes if type(only_classes) == torch.Tensor else torch.tensor(only_classes))
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, datasets.MNIST.classes


# Data transformations and loading - EMNIST
def get_emnist_data_loaders(batch_size=32, drop_last=True):
    DATA_PATH = os.path.join(os.getcwd(),"data")
    img_transformation = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(DATA_PATH, train=True, download=False, transform=img_transformation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    test_dataset = datasets.EMNIST(DATA_PATH, train=False, download=False, transform=img_transformation)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, datasets.EMNIST.classes


#     ### perform a few data tests
# def test_labels_shifted_by_one(data_loader):
#     # test that the labels are shifted by one to the right
#     for x, y in data_loader:
#         assert x.shape == (config["batch_size"], config["seq_len"], 784 + 10)
#         assert y.shape == (config["batch_size"], config["seq_len"])
#         assert torch.all(x[:,0,-10:] == 0)
#         assert torch.all(x[:,1:,-10:].argmax(-1) == y[:,:-1])
#         break

# def test_labels_not_shifted_by_one(data_loader):
#     # test that the labels are not shifted by one to the right
#     for x, y in data_loader:
#         assert x.shape == (config["batch_size"], config["seq_len"], 784 + 10)
#         assert y.shape == (config["batch_size"],)
#         assert torch.all(x[:,-1,-10:] == 0)
#         assert torch.all(x[:,:-1,-10:].max(-1).values == 1.)
#         break

# if __name__ == "__main__":
#     config = {
#     "epochs": 40,
#     "batch_size": 128,
#     "seq_len": 100,
#     "num_of_tasks": 2**16,
#     "permuted_images_frac": 1.0,
#     "permuted_labels_frac": 0.1,
#     # "whole_seq_prediction": True,
#     "whole_seq_prediction": False,
#     "lr": 3e-4,
#     "eps": 1e-16,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "ckpt_dir": "artifacts/models",
#     "ckpt_freq": 2,
#     "in_context_learner": {
#         "dim": 256,
#         "depth": 4,
#         "heads": 6,
#         "dim_head": 32,
#         "inner_dim": None, # fill in below
#         "dropout": 0.15,
#         "whole_seq_prediction": None, # fill in below
#     },
#     }
#     config["in_context_learner"]["inner_dim"] = config["in_context_learner"]["dim"] * 4
#     config["in_context_learner"]["whole_seq_prediction"] = config["whole_seq_prediction"]
#     rand_lin_proj_mnist_dataset_train = RandomLinearProjectionMNIST(
#     orig_mnist_dataset=datasets.MNIST(DATA_PATH, train=True, download=False, transform=RandomLinearProjectionMNIST.get_default_transform()),
#     num_tasks=config["num_of_tasks"],
#     seq_len=config["seq_len"],
#     labels_shifted_by_one=config["whole_seq_prediction"]
#     )
#     train_loader = torch.utils.data.DataLoader(rand_lin_proj_mnist_dataset_train, batch_size=config["batch_size"], num_workers=8, shuffle=True)

#     rand_lin_proj_mnist_dataset_test = RandomLinearProjectionMNIST(
#         orig_mnist_dataset=datasets.MNIST(DATA_PATH, train=False, download=False, transform=RandomLinearProjectionMNIST.get_default_transform()),
#         num_tasks=config["num_of_tasks"],
#         seq_len=config["seq_len"],
#         labels_shifted_by_one=config["whole_seq_prediction"]
#     )
#     test_loader = torch.utils.data.DataLoader(rand_lin_proj_mnist_dataset_test, batch_size=config["batch_size"],num_workers=8, shuffle=True)
    
#     if config["whole_seq_prediction"]:
#         test_labels_shifted_by_one(data_loader=train_loader)
#         test_labels_shifted_by_one(data_loader=test_loader)
#     else:
#         test_labels_not_shifted_by_one(data_loader=train_loader)
#         test_labels_not_shifted_by_one(data_loader=test_loader)

