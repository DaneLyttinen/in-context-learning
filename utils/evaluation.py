import numpy as np
import torch
import torch.nn.functional as F

def evaluate(model, test_loader, config):
    model.eval()
    loss, acc = 0.0, 0.0
    acc_over_seq = np.zeros(config["seq_len"])
    acc_max_improvement_within_seq = 0.0
    with torch.no_grad():
        # loss, acc, acc_max_improvement_within_seq = 0, 0, 0
        # acc_over_seq = np.array([0.] * config["seq_len"])
        for x, y in test_loader:
            x, y = x.to(config["device"], non_blocking=True), y.to(config["device"], non_blocking=True)

            y_hat = model(x)
            if config["whole_seq_prediction"]:
                loss += F.cross_entropy(y_hat.view(-1, 10), y.view(-1)).item()
                accuracy_per_seq = (y_hat.argmax(dim=-1) == y).float().mean(dim=0)
                acc_over_seq += accuracy_per_seq.cpu().numpy()
                acc_improvement = (y_hat[:, 1:, :].argmax(dim=-1) == y[:, 1:]).float().max(dim=1).values
                acc_degradation = (y_hat[:, 0, :].argmax(dim=-1) == y[:, 0]).float()
                acc_max_improvement_within_seq += (acc_improvement - acc_degradation).mean().item()
            else:
                loss += F.cross_entropy(y_hat, y).item()
            acc += (y_hat.argmax(dim=-1) == y).float().mean().item()
        num_batches = len(test_loader)
        loss /= num_batches
        acc /= num_batches
        acc_over_seq /= num_batches
        acc_max_improvement_within_seq /= num_batches
        print(f"loss: {loss:.4f}, acc: {acc:.4f}")
    return loss, acc, list(acc_over_seq), acc_max_improvement_within_seq