import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import f1_score

from dataset import BlinkDataset
import model as m


CONFIG = {
    'dataset': '/db/mEBAL/testdata.txt',
    'ckpt': './ckpt',
    'resize': (50, 50),
    'workers': 8,
    'batch_size': 256,
    'model': 'ResNet20'
}


def main():
    trainlst = CONFIG['dataset']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG['resize'])
    ])

    dataset = BlinkDataset(trainlst, transform)

    testloader = DataLoader(dataset,
                            batch_size=CONFIG['batch_size'],
                            shuffle=False,
                            num_workers=CONFIG['workers'],
                            pin_memory=True)

    model = getattr(m, CONFIG['model'])(3, 2)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    test_batches = len(testloader)
    print(f'TEST BATCHES: {test_batches}')

    best_ckpt = os.path.join(CONFIG['ckpt'], 'best.pth')

    state_dict = torch.load(best_ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    testloss = 0.0
    testf1 = 0.0
    total = 0
    correct = 0

    for data in testloader:
        with torch.no_grad():
            left_eyes, right_eyes, labels = data

            left_eyes = left_eyes.to(device)
            right_eyes = right_eyes.to(device)
            labels = labels.to(device)

            outputs = model(left_eyes, right_eyes)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            testf1 += f1_score(labels.cpu().numpy(),
                               predicted.cpu().numpy())

            loss = criterion(outputs, labels)
            testloss += loss.item()

    testloss /= test_batches
    acc = correct / total
    testf1 /= test_batches

    print_str = f'TEST LOSS: {testloss:.4f}, TEST ACCURACY: {acc:.3f} '
    print_str += f'TEST F1: {testf1:.3f}'
    print(print_str)


if __name__ == '__main__':
    main()
