import torch
from tqdm import tqdm
import time
from evaluate import evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, train_loader, validation_loader, optimizer, criterion, epochs, scheduler=None):
    
    epoch_times = []
    total_loss = []    
    model.train()
    
    accuracies_validation = []
    accuracies_train = []
    device = get_device()
    model = model.to(device)
    
    steps = 10

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_tic = time.time()
        for x, y, _ in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            
            yhat = model(x)
            
            loss = criterion(yhat, y)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        
        print('Evaluating epoch...', flush=True)
        test_accuracy = evaluate(model, validation_loader)
        train_accuracy = evaluate(model, train_loader)

        if scheduler != None:
            lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {lr}')
            scheduler.step()
        
        accuracies_validation.append(test_accuracy)
        accuracies_train.append(train_accuracy)

        total_loss.append(epoch_loss)
        
        epoch_toc = time.time()
        epoch_time = epoch_toc - epoch_tic
        epoch_times.append(epoch_time)

        print(f'Epoch: {epoch} | Loss: {epoch_loss:.2f} | Runtime: {epoch_time:.2f} seconds')
    
    return total_loss, epoch_times, accuracies_train, accuracies_validation


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device