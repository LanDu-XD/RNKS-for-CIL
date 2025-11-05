import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

EPSILON = 1e-8
batch_size = 128


class BaseLearner(object):
    def __init__(self):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._old_proxy = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': self._network.state_dict(),
        }
        torch.save(save_dict, '{}_{}.pkl'.format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top{}'.format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true),
                                                   decimals=2)

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (image, label, meta) in enumerate(loader):
            image, label = image.to(self._device), label.to(self._device)
            with torch.no_grad():
                feature = self._network.convnet(image, feature_class=True)
                outputs = self._network.fc(feature)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(label.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def incremental_train(self, data_manager):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader, mode='test'):
        model.eval()
        correct, total = 0, 0
        for i, (image, label, meta) in enumerate(loader):
            if mode=='train':
                image = image[1].to(self._device)
            else:
                image = image.to(self._device)
            with torch.no_grad():
                feature = model.convnet(image, feature_class=True)
                outputs = model.fc(feature)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == label).sum()
            total += len(label)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)




    ## NCM ##
    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _inputs, _targets, _meta in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self._network.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', cur_task=self._cur_task, appendent=(dd, dt))  # 获取当前削减后的存储样本的转换输入
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T  # 特征归一化
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',  # 返回的idx_dataset是get_dataset中经trsf的预处理样本，为了输入网络获取特征，并在原始data中返回需要保存的原始样本
                                                                  mode='test', cur_task=self._cur_task, ret_data=True)

            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T  # 特征归一化，消除量级影响
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection


            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets


            idx_dataset = data_manager.get_dataset([], source='train', mode='test', cur_task=self._cur_task,
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean


class FusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, output, label):
        length = output.shape[0]
        for i in range(length):
            self.matrix[output[i], label[i]] += 1

    def get_rec_per_class(self):
        rec = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        rec[np.isnan(rec)] = 0
        return rec

    def get_pre_per_class(self):
        pre = np.array(
            [
                self.matrix[i, i] / self.matrix[i, :].sum()
                for i in range(self.num_classes)
            ]
        )
        pre[np.isnan(pre)] = 0
        return pre

    def get_accuracy(self):
        acc = (
            np.sum([self.matrix[i, i] for i in range(self.num_classes)])
            / self.matrix.sum()
        )
        return acc

    def plot_confusion_matrix(self, normalize = False, cmap=plt.cm.Blues):

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes),
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig
