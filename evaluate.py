import torch
import tqdm


def evaluate(model, test_loader):
    
    device = get_device()
    model.eval()
    
    accuracy = 0
    num_correct = 0
    model = model.to(device)
    for x, y, _ in test_loader:
            
        x = x.to(device)
        y = y.to(device)
            
        yhat = model(x)
        _, label = torch.max(yhat, 1)
        num_correct += (y == label).sum().item()
            

    accuracy = 100 * num_correct / (len(test_loader) * test_loader.batch_size)
        
    
    return accuracy


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device