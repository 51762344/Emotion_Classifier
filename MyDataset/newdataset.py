import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


class EmotionDataset(Dataset):
    def __init__(self, images_path, labels, transform=None):
        self.images_path = images_path
        self.images_class = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, id):

        img = Image.open(self.images_path[id])
        img=img.convert('RGB')
        label = self.images_class[id]

        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def EmotionDataloader(path, labels, args, train=False):

    if train:
        shuffle = True
        data_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomRotation(8),
            transforms.CenterCrop([224, 224]),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            # transforms.FiveCrop(50),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        shuffle = False
        data_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    Dataset = EmotionDataset(path, labels,transform=data_transform)
    Dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    return Dataloader
