import torch
import tqdm


def evaluate(model, test_loader):
    
    device = get_device()
    model.eval()
    total_accuracy = []
    
    # for epoch in tqdm.tqdm(range(epochs)):
    #     num_correct = 0
    accuracy = 0
    num_correct = 0
    for x, y, _ in test_loader:
            
        x.to(device)
        y.to(device)
            
        yhat = model(x)
        _, label = torch.max(yhat, 1)
        num_correct += (y == label).sum().item()
            

        accuracy = 100 * num_correct / (len(test_loader) * test_loader.batch_size)
        total_accuracy.append(accuracy)
        
        # print(f'Epoch: {epoch} | Accuracy: {accuracy:.2f}%')
    
    return accuracy


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device