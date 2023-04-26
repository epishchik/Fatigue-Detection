import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import os

from dataset import BlinkDataset
import model as m


CONFIG = {
    'dataset': '/db/mEBAL/traindata.txt',
    'ckpt': './ckpt',
    'resize': (50, 50),
    'split': 0.8,
    'workers': 8,
    'batch_size': 128,
    'epochs': 20,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'model': 'ResNet20'
}


def main():
    trainlst = CONFIG['dataset']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CONFIG['resize']),
        transforms.ColorJitter(brightness=0.4,
                               saturation=0.1,
                               hue=0.2)
    ])

    dataset = BlinkDataset(trainlst, transform)
    trainsize = int(len(dataset) * CONFIG['split'])
    valsize = len(dataset) - trainsize
    trainset, valset = random_split(dataset, [trainsize, valsize])

    trainloader = DataLoader(trainset,
                             batch_size=CONFIG['batch_size'],
                             shuffle=True,
                             num_workers=CONFIG['workers'],
                             pin_memory=True)

    valloader = DataLoader(valset,
                           batch_size=CONFIG['batch_size'],
                           shuffle=True,
                           num_workers=CONFIG['workers'],
                           pin_memory=True)

    model = getattr(m, CONFIG['model'])(3, 2)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(model.parameters(),
                     lr=CONFIG['lr'],
                     weight_decay=CONFIG['weight_decay'])

    criterion = nn.CrossEntropyLoss()

    train_batches = len(trainloader)
    val_batches = len(valloader)
    print(f'TRAIN BATCHES: {train_batches}, VAL BATCHES: {val_batches}')

    try:
        ckpt = CONFIG['ckpt']
        os.system(f'mkdir {ckpt}')
    except Exception as _:
        pass

    last_ckpt = os.path.join(CONFIG['ckpt'], 'last.pth')
    best_ckpt = os.path.join(CONFIG['ckpt'], 'best.pth')

    bestloss = 9999999999
    for epoch in range(CONFIG['epochs']):
        model.train()
        trainloss = 0.0

        for data in trainloader:
            left_eyes, right_eyes, labels = data

            left_eyes = left_eyes.to(device)
            right_eyes = right_eyes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(left_eyes, right_eyes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            trainloss += loss.item()

        valloss = 0.0
        total = 0
        correct = 0
        model.eval()

        for data in valloader:
            with torch.no_grad():
                left_eyes, right_eyes, labels = data

                left_eyes = left_eyes.to(device)
                right_eyes = right_eyes.to(device)
                labels = labels.to(device)

                outputs = model(left_eyes, right_eyes)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                valloss += loss.item()

        trainloss /= train_batches
        valloss /= val_batches
        acc = correct / total

        if valloss < bestloss:
            bestloss = valloss
            torch.save(model.state_dict(), best_ckpt)
        torch.save(model.state_dict(), last_ckpt)

        print_str = f'EPOCH: {epoch + 1}, TRAIN LOSS: {trainloss:.4f}, '
        print_str += f'VAL LOSS: {valloss:.4f}, VAL ACCURACY: {acc:.3f}'
        print(print_str)


if __name__ == '__main__':
    main()
