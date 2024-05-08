import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import tqdm
#import wandb
from utils.data import RandomLinearProjectionMNIST, DATA_PATH
from utils.early_stopper import EarlyStopper
from utils.io import save_checkpoint
from model import InContextLearner
from torchvision import datasets
from utils.evaluation import evaluate

global config
config = {
    "epochs": 40,
    "batch_size": 256,
    "seq_len": 100,
    "num_of_tasks": 2**16,
    "permuted_images_frac": 1.0,
    "permuted_labels_frac": 0.1,
    # "whole_seq_prediction": True,
    "whole_seq_prediction": False,
    "lr": 3e-4,
    "eps": 1e-16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir": "artifacts/models",
    "ckpt_freq": 2,
    "in_context_learner": {
        "dim": 256,
        "depth": 4,
        "heads": 6,
        "dim_head": 32,
        "inner_dim": None, # fill in below
        "dropout": 0.15,
        "whole_seq_prediction": None, # fill in below
    },
}
config["in_context_learner"]["inner_dim"] = config["in_context_learner"]["dim"] * 4
config["in_context_learner"]["whole_seq_prediction"] = config["whole_seq_prediction"]

#print(f"... Running on {config['device']} ...")

def init_dicts(train_results, test_results, hidden_size, num_tasks):
    train_results[hidden_size][num_tasks] = {}
    test_results[hidden_size][num_tasks] = {}
    train_results[hidden_size][num_tasks]["loss"] = []
    train_results[hidden_size][num_tasks]["accuraccy"] = []
    test_results[hidden_size][num_tasks]["loss"] = []
    test_results[hidden_size][num_tasks]["accuraccy"] = []

def train_model(hidden_size):
    num_tasks_range = [2**i for i in range(4, 24)]  # Range of number of tasks

    train_results = {}
    train_results[hidden_size] = {}
    test_results = {}
    test_results[hidden_size] = {}
    #for hidden_size in hidden_size_range:
    for num_tasks in num_tasks_range:
        init_dicts(train_results,test_results, hidden_size,num_tasks)
        config["num_of_tasks"] = num_tasks
        config["in_context_learner"]["dim"] = hidden_size
        config["in_context_learner"]["inner_dim"] = config["in_context_learner"]["dim"] * 4
        model = InContextLearner(**config["in_context_learner"]).to(config["device"])
        model_optim = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"])
        if num_tasks != num_tasks_range[0]:
            del train_loader, test_loader
        train_loader,test_loader = init_datasets(hidden_size)
        early_stopper = EarlyStopper(patience=4, min_delta=0.04)
        print(f"num_tasks {num_tasks}, hidden size {hidden_size}")
        for epoch in tqdm.tqdm(range(config["epochs"])):
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(config["device"]), y.to(config["device"])
                y_hat = model(x)
                if config["whole_seq_prediction"]:
                    loss = F.cross_entropy(y_hat.view(-1, 10), y.view(-1))
                else:
                    loss = F.cross_entropy(y_hat, y)
                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
                accuracy = (y_hat.argmax(dim=-1) == y).float().mean().item()
                print(f"Training loss: {loss.item()}")
                print(f"Training accuracy {accuracy}")
                train_results[hidden_size][num_tasks]["loss"].append(loss.item())
                train_results[hidden_size][num_tasks]["accuraccy"].append(accuracy)
                
            ### evaluate
            eval_loss, eval_acc, acc_over_seq, acc_max_improvement_within_seq = evaluate(model, test_loader, config)
            #live_plot.update({"eval_loss": eval_loss, "eval_acc": eval_acc})
            if config["whole_seq_prediction"]:
                print(acc_max_improvement_within_seq)
                print(acc_over_seq)
            print(f"eval loss {eval_loss}")
            print(f"eval accuracy {eval_acc}")
            test_results[hidden_size][num_tasks]["loss"].append(eval_loss)
            test_results[hidden_size][num_tasks]["accuraccy"].append(eval_acc)
            ### save model
            if epoch % config["ckpt_freq"] == 0:
               save_checkpoint(epoch=epoch, model=model, optimizer=model_optim, loss_train=loss.detach(), loss_eval=eval_loss, config=config)
               save_results(test_results,train_results,epoch,hidden_size, num_tasks)
            if early_stopper.early_stop(eval_loss):
                print(f"Stopping early")
                save_checkpoint(epoch=epoch, model=model, optimizer=model_optim, loss_train=loss.detach(), loss_eval=eval_loss, config=config)
                save_results(test_results,train_results,epoch,hidden_size, num_tasks)
                break
        save_results(test_results,train_results,epoch,hidden_size, num_tasks)
    return test_results, train_results

    #raise NotImplementedError
def save_results(test, train, epoch, hidden_size, num_tasks):
    filename = f"data/results/transformer_test_{epoch}_{hidden_size}_{num_tasks}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(test, file)
    filename = f"data/results/transformer_train_{epoch}_{hidden_size}_{num_tasks}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(train, file)

def init_datasets(hidden_size):
    rand_lin_proj_mnist_dataset_train = RandomLinearProjectionMNIST(
        orig_mnist_dataset=datasets.MNIST(DATA_PATH, train=True, download=False, transform=RandomLinearProjectionMNIST.get_default_transform()),
        num_tasks=config["num_of_tasks"],
        seq_len=config["seq_len"],
        hidden_size=hidden_size,
        labels_shifted_by_one=config["whole_seq_prediction"]
    )
    train_loader = torch.utils.data.DataLoader(rand_lin_proj_mnist_dataset_train, batch_size=config["batch_size"])

    rand_lin_proj_mnist_dataset_test = RandomLinearProjectionMNIST(
        orig_mnist_dataset=datasets.MNIST(DATA_PATH, train=False, download=False, transform=RandomLinearProjectionMNIST.get_default_transform()),
        num_tasks=config["num_of_tasks"],
        seq_len=config["seq_len"],
        hidden_size=hidden_size,
        labels_shifted_by_one=config["whole_seq_prediction"],
        is_train=False
    )
    test_loader = torch.utils.data.DataLoader(rand_lin_proj_mnist_dataset_test, batch_size=config["batch_size"])
    print("... Initialized datasets ...")
    return train_loader,test_loader
    ### save dataloaders locally
    # torch.save(train_loader, "artifacts/data/train_loader.pt")
    # torch.save(test_loader, "artifacts/data/test_loader.pt")

if __name__ == "__main__":
    hidden_size_range = [2**i for i in range(7, 10)]
    train_results_list = []
    test_results_list = []
    try:
        for hidden_size in hidden_size_range:
            test_results, train_results = train_model(hidden_size)
            train_results_list.append(train_results)
            test_results_list.append(test_results)
    except Exception as e:
        print(train_results_list)
        print(test_results_list)
        print(e)
    all_train_results = {}
    for result in train_results_list:
        all_train_results.update(result)
    print(all_train_results)
    all_test_results = {}
    for result in test_results_list:
        all_test_results.update(result)
    print(all_test_results)

    filename = 'transformers_results_train_1.pkl'
    # Writing the dictionary to a file
    with open(filename, 'wb') as file:
        pickle.dump(all_train_results, file)

    filename = 'transformers_results_test_1.pkl'
    # Writing the dictionary to a file
    with open(filename, 'wb') as file:
        pickle.dump(all_test_results, file)