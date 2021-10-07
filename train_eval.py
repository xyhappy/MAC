import time
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold

def cross_validation_with_val_set(dataset, model, seed, folds, lr, weight_decay,
                                  batch_size, epochs, device, patience, logger=None):

    test_accs, durations = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds, seed))):

        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        val_dataset = [dataset[i] for i in val_idx]

        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t_start = time.perf_counter()

        min_loss = 1e10
        max_patience = 0
        best_epoch = 0
        fold_test_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(model, optimizer, train_loader, device)
            val_loss, val_acc = test(model, val_loader, device)
            test_loss, test_acc = test(model, test_loader, device)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }

            logger(eval_info)

            if val_loss < min_loss:
                print("Model saved at epoch {}".format(epoch))
                min_loss = val_loss
                max_patience = 0
                best_epoch = epoch
                fold_test_acc = test_acc
            else:
                max_patience += 1
            if max_patience > patience:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        print("For fold {}, best epoch: {}, test acc: {:.6f}".format(
            fold+1, best_epoch, fold_test_acc))
        test_accs.append(fold_test_acc)

    test_acc, duration = torch.tensor(test_accs), torch.tensor(durations)
    acc_mean = test_acc.mean().item()
    acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()
    print('Test Accuracy: {:.6f} ± {:.6f}, Duration: {:.6f}'.format(acc_mean, acc_std, duration_mean))

    return acc_mean, acc_std, duration_mean

def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.graph_labels):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices

def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    train_loss = 0.
    train_correct = 0.
    num_graphs = 0
    for batch in trainloader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = out.argmax(dim=1)
        train_correct += pred.eq(batch_labels).sum().item()
        train_loss += F.nll_loss(out, batch_labels, reduction='sum').item()
        torch.cuda.empty_cache()
    return train_loss / num_graphs, train_correct / num_graphs

@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.
    loss = 0.
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs)
        pred = out.argmax(dim=1)
        correct += pred.eq(batch_labels).sum().item()
        loss += F.nll_loss(out, batch_labels, reduction='sum').item()
        torch.cuda.empty_cache()
    return loss / num_graphs, correct / num_graphs

