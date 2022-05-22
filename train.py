import torch
from tqdm import tqdm
import time
from evaluate import evaluate

def train(model, train_loader, test_loader, optimizer, criterion, epochs):
    
    epoch_times = []
    total_loss = []    
    model.train()
    
    accuracies_test = []
    accuracies_train = []
    device = get_device()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_tic = time.time()
        for x, y, _ in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            x.to(device)
            y.to(device)
            
            yhat = model(x)
            
            loss = criterion(yhat, y)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        test_accuracy = evaluate(model, test_loader)
        train_accuracy = evaluate(model, train_loader)

        accuracies_test.append(test_accuracy)
        accuracies_train.append(train_accuracy)

        print(f'Epoch: {epoch} | Loss: {epoch_loss:.2f}')
        total_loss.append(epoch_loss)
        
        epoch_toc = time.time()
        epoch_time = epoch_toc - epoch_tic
        print(f'Epoch: {epoch} took: {epoch_time:.2f} seconds')
        epoch_times.append(epoch_time)
    
    return total_loss, epoch_times, accuracies_train, accuracies_test


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device