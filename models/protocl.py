import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base_protocl import BaseLearner
from utils.inc_net import IncrementalNetWithPROTOCL
from convs.CL.lr_scheduler import WarmupMultiStepLR
from convs.CL.evaluate import accuracy, AverageMeter
import scipy.io as io
from convs.CL.proxynca import ProxyNCA
from convs.CL.cc import CC
import math
import os

# CIFAR100, ResNet32, 10 base
epochs = 200
lrate = 0.5
proxy_lrate = 1.0
milestones = [120, 160]
lrate_decay = 0.1
batch_size = 128
memory_size = 2000
T = 2


class PROTOCL(BaseLearner):
    def __init__(self, args):
        super().__init__()
        self._device = args['device']
        self._network = IncrementalNetWithPROTOCL(args['convnet_type'], self._device, epochs, False)
        self._criterion_cl = ProxyNCA(self._device, sz_embedding=256)
        self._nb_proxy = 6
        self._class_means = None
        self.show_step = 100

    def after_task(self):
        # self.save_checkpoint(logfilename)
        self._old_network = self._network.copy().freeze()
        # self._old_proxy = self._criterion_cl.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_fc(self._total_classes)
        self._criterion_cl.update_proxy(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train', cur_task=self._cur_task, appendent=self._get_memory())
        logging.info('train dset: {}'.format(len(train_dset)))
        path1 = os.path.join("task_{}traindata.pth".format(self._cur_task))
        torch.save(train_dset, path1)

        self.lamda = self._known_classes / self._total_classes

        logging.info('Lambda: {:.3f}'.format(self.lamda))
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test', cur_task=self._cur_task)

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        self._training(self.train_loader, self.test_loader)  # 执行训练

        # Exemplars
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)



    def _run(self, train_loader, test_loader, optimizer, scheduler):
        best_result, best_epoch = 0, 0

        self.epoch_loss = []
        self.epoch_protoclloss = []
        self.epoch_disloss = []
        self.epoch_ccdloss = []
        self.trainacc_list = []
        self.testacc_list = []
        self.func = torch.nn.Softmax(dim=1)
        CCD_loss = CC(0.4, 2, self._device)
        for epoch in range(1, epochs+1):
            self._network.train()
            all_loss = AverageMeter()
            protocl_loss = AverageMeter()
            dis_loss = AverageMeter()
            corr_loss = AverageMeter()
            acc = AverageMeter()
            self._network.reset_epoch(epoch)

            for i, (inputs, targets, meta) in enumerate(train_loader):

                targets = torch.cat([targets, targets], dim=0)
                inputs = torch.cat([inputs[0], inputs[1]], dim=0)
                cnt = targets.shape[0]
                inputs, targets, inputs_meta, label_meta = inputs.to(self._device), targets.to(self._device), meta["sample_image"].to(self._device), meta["sample_label"].to(self._device)  # 获取随机采样批次数据和逆采样批次数据

                logits, cl_features, l = self._network(inputs, meta)


                hybrid_loss = l * self._criterion_cl(cl_features, targets, mode='train') + (1 - l) * F.cross_entropy(logits, label_meta)  # 最终loss是两支路loss的加权输出
                now_result = torch.argmax(self.func(logits), 1)
                now_acc = accuracy(now_result.cpu().numpy(), label_meta.cpu().numpy())[0]

                if self._old_network is not None:
                    fea_s = self._network.convnet(inputs, feture_class=True)
                    old_logits, cl_features_old, _ = self._old_network(inputs, meta)
                    fea_t = self._old_network.convnet(inputs, feture_class=True)
                    fea_t = fea_t.detach()

                    CCDloss = CCD_loss(fea_s, fea_t, targets, self._known_classes)

                    old_logit = old_logits.detach()
                    hat_pai_k = F.softmax(old_logit / T, dim=1)
                    log_pai_k = F.log_softmax(logits[:, :self._known_classes] / T, dim=1)
                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                    ##########
                    loss = l * self._criterion_cl(cl_features, targets, mode='train') + 2*CCDloss + (1 - l) * (distill_loss * T * self.lamda + F.cross_entropy(logits, label_meta) * (1 - self.lamda))
                else:
                    loss = hybrid_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                all_loss.update(loss.data.item(), cnt)
                protocl_loss.update(hybrid_loss.data.item(), cnt)
                if self._old_network is not None:
                    dis_loss.update(distill_loss.data.item(), cnt)
                    corr_loss.update(CCDloss.data.item(), cnt)
                acc.update(now_acc, cnt)  # 记录当前批数据的acc信息

                if i % self.show_step == 0:
                    pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                        epoch, i, len(train_loader), loss.data.item(), acc.val * 100
                    )
                    logging.info(pbar_str)

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader, mode='train')
            test_acc = self._compute_accuracy(self._network, test_loader, mode='test')
            epoch_loss = all_loss.avg
            self.epoch_loss.append(epoch_loss)
            self.epoch_protoclloss.append(protocl_loss.avg)
            self.epoch_disloss.append(dis_loss.avg)
            self.epoch_ccdloss.append(corr_loss.avg)
            self.trainacc_list.append(train_acc)
            self.testacc_list.append(test_acc)
            if self._old_network is not None:
                info = '{} => Task {}, Epoch {}/{} => Loss {:.3f}, protoclLoss {:.3f}, disLoss {:.3f}, CCDLoss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    'train', self._cur_task, epoch, epochs, all_loss.avg, protocl_loss.avg, dis_loss.avg, corr_loss.avg, train_acc, test_acc)
            else:
                info = '{} => Task {}, Epoch {}/{} => Loss {:.3f}, protoclLoss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                        'train', self._cur_task, epoch, epochs, all_loss.avg, protocl_loss.avg, train_acc, test_acc)
            logging.info(info)
            if test_acc > best_result:
                best_result, best_epoch = test_acc, epoch
            logging.info(
                "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                    best_epoch, best_result
                )
            )


    def _training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        self._criterion_cl.to(self._device)
        # train network
        optimizer = torch.optim.SGD([{"params": self._network.parameters(), "lr": lrate},
                                     {"params": self._criterion_cl.parameters(), "lr": proxy_lrate}],
                                    lr=0.5, momentum=0.9,
                                    weight_decay=1e-4)

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lrate_decay, warmup_epochs=5,)
        self._run(train_loader, test_loader, optimizer, scheduler)

