import torch
import os

def get_y_pred_truth(model, data_loader):
    device = get_device()

    y_pred = []
    y_truth = []

    model.eval()
    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)
        
        yhat = model(x)
        
        _, label = torch.max(yhat, 1)
        y_pred.append(label.item())
        y_truth.append(y.item())

    return y_pred, y_truth

def get_device():
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    return device


def save_model(model, path):
    with open(path, 'wb') as file:
        torch.save({'model_state_dict': model.state_dict()}, file)


def load_model_state_dict(path):
    path = os.path.join(path, param_name)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    
    return model_state_dict