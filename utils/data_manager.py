import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet1000
import random

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, cur_task, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
            y = y.astype(np.int64)
        elif source == 'test':
            x, y = self._test_data, self._test_targets
            y = y.astype(np.int64)
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            trsf_cl = transforms.Compose([*self._train_cl_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
            trsf_cl = transforms.Compose([])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
            trsf_cl = transforms.Compose([])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, trsf_cl, mode, cur_task, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, trsf_cl, mode, cur_task, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._train_cl_trsf = idata.train_cl_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, cl_trsf, mode, cur_task, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.cl_trsf = cl_trsf
        self.use_path = use_path
        self.cur_task = cur_task
        self.train = True if mode == 'train' else False
        self.cls_num = len(np.unique(self.labels))
        if self.train:
            self.class_dict = self._get_class_dict()

    def __len__(self):
        return len(self.images)

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def __getitem__(self, idx):

        img, target = self.images[idx], self.labels[idx]
        meta = dict()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.train:
            if self.cur_task >= 1:
                sample_class = random.randint(0, self.cls_num - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cur_task == 0:
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.images[sample_index], self.labels[sample_index]
            sample_img = Image.fromarray(sample_img)
            sample_img = self.trsf(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

            img = [self.trsf(img), self.cl_trsf(img)]
        elif self.trsf is not None:
            img = self.trsf(img)

        return img, target, meta


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar100':
        return iCIFAR100()
    elif name == 'imagenet1000':
        return iImageNet1000()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
