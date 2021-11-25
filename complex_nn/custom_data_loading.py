import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pickle


class MyDataset(Dataset):
    def __init__(self, data_folder, one_file, n_classes):

        self.X = []
        self.Y = []
        self.n_classes = n_classes
        self.X_v = []
        self.metadata = []
        self.exps = []
        if one_file:
            with open(data_folder + '.pkl', 'rb') as f:
                tmp = pickle.load(f)
                for a in tmp:
                    self.X.append(tmp[a][0])
                    self.X_v.append(tmp[a][1])
                    self.Y.append(int(a.split('_c')[0]))
                    self.metadata.append(tmp[a][2])
                    self.exps.append(a.split('_')[1])
        else:
            for f in os.listdir(data_folder):
                if f.endswith('npy'):
                    tmp = np.load(f'{data_folder}/{f}', allow_pickle=True)
                    self.X.append(tmp[0])
                    self.X_v.append(tmp[1])  ##usually 1Hz data
                    self.Y.append(int(f.split('_c')[0]))  ##labels are encoded in the filename
                    self.metadata.append(tmp[2])
                    self.exps.append(f.split('_')[1])
        self.X = np.stack(self.X)
        if len(self.X_v[0]) > 0:
            self.X_v = np.stack(self.X_v)
            self.dim_v = self.X_v.shape[1]
        else:
            ##1Hz data are not mandatory
            self.X_v = np.zeros(self.X.shape[0])
            self.dim_v = 0

        self.Y = np.array(self.Y)
        self.classes = np.unique(self.Y)

        self.nrow = self.X.shape[2]
        self.ncol = self.X.shape[3]
        self.n_channels = self.X.shape[1]
        print(f'    loaded {self.X.shape[0]} samples')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idxs):
        x = self.X[idxs]
        #x_v = self.X_v[idxs]
        y = self.Y[idxs]
        one_hot_y = np.eye(self.n_classes)[y]
        return [x, one_hot_y]
        #return [[x, x_v], y]


##What follows is based on https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA
def load_data(data_folder, batch_size, train, n_classes=2, num_workers=0, infinite_data_loader=False, one_file=True):
    data = MyDataset(data_folder, one_file=one_file, n_classes=n_classes)
    data_loader = get_data_loader(data,
                                  batch_size=batch_size,
                                  shuffle=True if train else False,
                                  num_workers=num_workers,
                                  infinite_data_loader=infinite_data_loader,
                                  drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class, data.dim_v, data.nrow, data.ncol, data.n_channels


def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers)


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights, replacement=False, num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=False)

        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=_InfiniteSampler(batch_sampler)))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0  # Always return 0


def get_data(path, batch_size, num_workers, one_file, n_classes):

    folder_source_train = os.path.join(path, 'source_train')
    folder_target_train = os.path.join(path, 'target_train')
    folder_source_test = os.path.join(path, 'source_test')
    folder_target_test = os.path.join(path, 'target_test')
    print('Source_train_loader...')
    source_train_loader, n_class, dim_v, nrow, ncol, n_channels = load_data(folder_source_train,
                                                                            batch_size,
                                                                            n_classes=n_classes,
                                                                            infinite_data_loader=True,
                                                                            train=True,
                                                                            num_workers=num_workers,
                                                                            one_file=one_file)
    print('Target_train_loader...')
    target_train_loader, _, _, _, _, _ = load_data(folder_target_train,
                                                   batch_size,
                                                   n_classes=n_classes,
                                                   infinite_data_loader=True,
                                                   train=True,
                                                   num_workers=num_workers,
                                                   one_file=one_file)
    print('Source_test_loader...')
    source_test_loader, _, _, _, _, _ = load_data(folder_source_test,
                                                  batch_size,
                                                  n_classes=n_classes,
                                                  infinite_data_loader=False,
                                                  train=False,
                                                  num_workers=num_workers,
                                                  one_file=one_file)
    print('Target_test_loader...')
    target_test_loader, _, _, _, _, _ = load_data(folder_target_test,
                                                  batch_size,
                                                  n_classes=n_classes,
                                                  infinite_data_loader=False,
                                                  train=False,
                                                  num_workers=num_workers,
                                                  one_file=one_file)

    source_train_loader_test, _, _, _, _, _ = load_data(folder_source_train,
                                                        batch_size,
                                                        n_classes=n_classes,
                                                        infinite_data_loader=False,
                                                        train=False,
                                                        num_workers=num_workers,
                                                        one_file=one_file)
    target_train_loader_test, _, _, _, _, _ = load_data(folder_target_train,
                                                        batch_size,
                                                        n_classes=n_classes,
                                                        infinite_data_loader=False,
                                                        train=False,
                                                        num_workers=num_workers,
                                                        one_file=one_file)
    return source_train_loader, target_train_loader, source_test_loader, target_test_loader, source_train_loader_test, target_train_loader_test, n_class, dim_v, nrow, ncol, n_channels
