import os
import torch


def save_checkpoint(epoch, model, optimizer, loss_train, loss_eval, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_train": loss_train,
        "loss_eval": loss_eval,
        "config": config,
    }, os.path.join(config["ckpt_dir"], f"model_{epoch}_{config['in_context_learner']['dim']}_{config['num_of_tasks']}.pt"))

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss_train = checkpoint["loss_train"]
    loss_eval = checkpoint["loss_eval"]
    config = checkpoint["config"]
    return model, optimizer, epoch, loss_train, loss_eval, config