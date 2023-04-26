from torch.utils.data import Dataset
import cv2


class BlinkDataset(Dataset):
    def __init__(self, file, transform, sep='$'):
        self.samples = []
        self.transform = transform

        with open(file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                sample = line.strip().split(sep)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        left_eye = self.transform(cv2.imread(sample[0]))
        right_eye = self.transform(cv2.imread(sample[1]))

        label = int(sample[2])

        return left_eye, right_eye, label
