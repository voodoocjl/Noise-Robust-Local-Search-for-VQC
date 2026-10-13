import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders, qml_Dataloaders
from FusionModel import QNet
from FusionModel import translator, single_enta_to_design

from Arguments import Arguments
import random
from tqdm import tqdm

def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)


def compute_margin_hinge_loss(log_probs, targets, margin_target):
    """Compute hinge loss on true-vs-strongest-wrong class margin using log-prob scores."""
    if log_probs.ndim != 2:
        raise ValueError(f"Expected log_probs to be 2D [batch, classes], got shape {tuple(log_probs.shape)}")

    num_classes = log_probs.size(1)
    if num_classes < 2:
        return torch.tensor(0.0, device=log_probs.device, dtype=log_probs.dtype)

    true_scores = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
    wrong_scores = log_probs.clone()
    wrong_scores.scatter_(1, targets.view(-1, 1), float('-inf'))
    max_wrong_scores, _ = wrong_scores.max(dim=1)
    margins = true_scores - max_wrong_scores
    margin_target_tensor = torch.tensor(margin_target, device=log_probs.device, dtype=log_probs.dtype)
    return torch.clamp(margin_target_tensor - margins, min=0.0).mean()

    
def train(model, data_loader, optimizer, criterion, args, margin_enabled=False, margin_weight=0.0, margin_target=0.0):
    model.train()
    for feed_dict in tqdm(data_loader, desc="Training", disable=True):
    # for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)    
        optimizer.zero_grad()
        output = model(images, args.n_qubits, args.task)
        loss = criterion(output, targets)
        if margin_enabled and margin_weight > 0:
            margin_loss = compute_margin_hinge_loss(output, targets, margin_target)
            loss = loss + margin_weight * margin_loss
        loss.backward()
        optimizer.step()

def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor().to(args.device)
    output_all = torch.Tensor().to(args.device)
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images, args.n_qubits, args.task)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
            output_all = torch.cat((output_all, output), dim=0) 
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    
    return total_loss, accuracy

def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    if args.backend == 'qi':
        tqdm_disable = False
    else:
        tqdm_disable = True
    
    with torch.no_grad():
        for feed_dict in tqdm(data_loader, desc="Evaluating", disable=tqdm_disable):
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)        
            output = model(images, args.n_qubits, args.task)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy
    # metrics = output.cpu().numpy()
    return metrics


def evaluate_margin(model, data_loader, args):
    """Return mean top1-top2 log-prob margin over a data loader."""
    model.eval()
    margins = []
    if args.backend == 'qi':
        tqdm_disable = False
    else:
        tqdm_disable = True

    with torch.no_grad():
        for feed_dict in tqdm(data_loader, desc="Evaluating Margin", disable=tqdm_disable):
            images = feed_dict['image'].to(args.device)
            output = model(images, args.n_qubits, args.task)
            if output.ndim != 2 or output.size(1) < 2:
                continue
            top2_values, _ = output.topk(2, dim=1)
            batch_margin = top2_values[:, 0] - top2_values[:, 1]
            margins.append(batch_margin)

    if not margins:
        return None

    all_margins = torch.cat(margins, dim=0)
    return float(all_margins.mean().item())

def Scheme_eval(design, task, weight, noise=None):
    result = {}  
    args = Arguments(**task)
    if noise:
        args.backend = 'qi'
    path = 'weights/'  
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'])
   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(torch.load(path+weight), strict= False)
    result['mae'] = evaluate(model, test_loader, args)
    result['margin'] = evaluate_margin(model, test_loader, args)
    return model, result

def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        args = Arguments(**task)
        if epochs == None:
            epochs = args.epochs
        
        if task['task'].startswith('QML'):
            dataloader = qml_Dataloaders(args)
        else:
            dataloader = MNISTDataLoaders(args, task['task'])
       
        train_loader, val_loader, test_loader = dataloader
        model = QNet(args, design).to(args.device)
        if weight != 'init':
            if weight != 'base':
                model.load_state_dict(weight, strict= False)
            else:            
                model.load_state_dict(torch.load('init_weights/base_fashion'))

        margin_enabled = bool(task.get('margin_enabled', False))
        margin_weight = float(task.get('margin_weight', 0.2))
        margin_target = float(task.get('margin_target', 0.3))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
        train_loss_list, val_loss_list = [], []
        best_val_loss = 0
        start = time.time()
        best_model = model   

        if epochs == 0:
            print('No training epochs specified, skipping training.')
        else:
            for epoch in range(epochs):
                try:
                    train(
                        model,
                        train_loader,
                        optimizer,
                        criterion,
                        args,
                        margin_enabled=margin_enabled,
                        margin_weight=margin_weight,
                        margin_target=margin_target,
                    )
                except Exception as e:
                    print('No parameter gate exists')
                train_loss = test(model, train_loader, criterion, args)
                train_loss_list.append(train_loss)        
                val_loss = evaluate(model, val_loader, args)
                val_loss_list.append(val_loss)
                metrics = evaluate(model, test_loader, args)
                val_loss = 0.5 *(val_loss+train_loss[-1])
                if val_loss > best_val_loss:
                    best_val_loss = val_loss
                    if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics, 'saving model')
                    best_model = copy.deepcopy(model)           
                else:
                    if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics)        
        end = time.time()        
        metrics = evaluate(best_model, test_loader, args)
        margin = evaluate_margin(best_model, test_loader, args)
        display(metrics)
        print("Running time: %s seconds" % (end - start))
        report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'mae': metrics, 'margin': margin}
        
        if save:
            torch.save(best_model.state_dict(), 'init_weights/init_weight_' + task['task'])
        return best_model, report
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

def pretrain(design, task, weight):    

    args = Arguments(**task)
    
    if task['task'].startswith('QML'):
        dataloader = qml_Dataloaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task['task'])   
    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    model.load_state_dict(weight, strict= True)
    
    val_loss = evaluate(model, val_loader, args)
    display(val_loss)
    
    return val_loss


if __name__ == '__main__':
    task = {
    'task': 'MNIST_10',
    'option': 'mix_reg',
    'n_qubits': 10,
    'n_layers': 4,
    'fold': 1
    }

    # task = {
    # 'task': 'MNIST_4',
    # 'option': 'mix_reg',
    # 'n_qubits': 4,
    # 'n_layers': 4,
    # 'fold': 1,
    # 'backend': 'tq'
    # }

    # task = {
    # 'task': 'QML_Hidden_80d',
    # 'n_qubits': 20,
    # 'n_layers': 4,
    # 'fold': 5,
    # 'option': 'mix_reg',
    # 'regular': True,
    # 'num_processes': 2
    # }
    
    arch_code = [task['n_qubits'], task['n_layers']]
    args = Arguments(**task)
    n_layers = arch_code[1]
    n_qubits = int(arch_code[0] / args.fold)
    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]

    # single = [[5, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1], [2, 0, 1, 1, 1, 1, 1, 0, 1], [3, 0, 1, 1, 1, 1, 1, 1, 1], [4, 0, 1, 1, 1, 0, 1, 1, 1]]
    # enta =  [[1, 2, 2, 3, 2], [2, 1, 3, 3, 5], [3, 2, 2, 1, 4], [4, 1, 1, 2, 2], [5, 1, 2, 4, 4]]

    
    # design = translator(single, enta, 'full', arch_code, args.fold)
    # design = op_list_to_design(op_list, arch_code)
    design = single_enta_to_design(single, enta, arch_code, args.fold)

    best_model, report = Scheme(design, task, 'init', 30, verbs=False, save=False)
    # torch.save(best_model.state_dict(), 'weights/tmp')
   
    # result = Scheme_eval(design, task, 'tmp_4', noise=False)
    # display(result)
    # result = Scheme_eval(design, task, 'tmp_4', noise=True)
    # display(result)

    # torch.save(best_model.state_dict(), 'weights/base_fashion')