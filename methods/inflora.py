import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_inflora import SiNet
from models.vit_inflora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import math

from optimer_sam import SAM, enable_running_stats, disable_running_stats

class InfLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        #  为 SAM 优化器设置 rho
        self.rho = args["rho"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # if self._old_network is not None:
        #     self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            # 先冻结所有参数
            param.requires_grad_(False)
            try:
                # 只训练最后一层 任务分类器
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                # LoRA分支的 B矩阵的 value 和 key 部分
                # numtask - 1 就是当前任务编号（因为从0开始
                # 所以当检查 name 时，会找到 "lora_B_k.0"、"lora_B_k.1" 等，
                # 只对当前任务编号对应的参数设置为 requires_grad=True，其他全部冻结
                if "lora_B_k" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        # 在无梯度模式下，提取当前任务的输入特征用于构造 cur_matrix
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)
                # if i > 3: break

            if self._cur_task == 0:
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix

                        # 直接对 cur_matrix 做 SVD，提取主成分方向（子空间）作为新的 LoRA 分支初始值
                        # 1/√3 是一个缩放因子，常用于初始化稳定训
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:  # 后续任务

                # kk = 0
                # for module in self._network.modules():
                #     if isinstance(module, Attention_LoRA):
                #         cur_matrix = module.cur_matrix
                #         cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                #         cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                #         module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.cur_matrix.zero_()
                #         module.n_cur_matrix = 0
                #         kk += 1

                kk = 0
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix

                        # 对 cur_matrix 进行子空间投影
                        
                        # remove：去除旧任务子空间影响（从 cur_matrix 中减去
                        if self.project_type[kk] == 'remove':
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                        # retain：保留旧任务主方向（映射到旧子空间
                        else:
                            assert self.project_type[kk] == 'retain'
                            cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                        
                        # 直接对 cur_matrix 做 SVD，提取主成分方向（子空间）作为新的 LoRA 分支初始值
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)

                        # 1/√3 是一个缩放因子，常用于初始化稳定训
                        module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))


                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

        print(f"Parameters to be updated: {enabled}")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        #  Optimizer 构造与主训练
        # 区分首个任务和后续任务的训练轮数与学习率配置（首任务通常训练更充分）
        if self._cur_task==0:
            print(f'Debug init task self.optim: {self.optim}')
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            elif self.optim == 'sam-sgd':
                # optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                optimizer = SAM(
                    self._network.parameters(),
                    base_optimizer=optim.SGD,
                    rho=self.rho,
                    lr=self.init_lr,
                    momentum=0.9,
                    weight_decay=self.init_weight_decay
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer.base_optimizer,T_max=self.init_epoch)

            elif self.optim == 'sam-adam':
                # optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                
                optimizer = SAM(
                    self._network.parameters(),
                    base_optimizer=optim.Adam,
                    rho=self.rho,
                    lr=self.init_lr,
                    betas=(0.9,0.999),
                    weight_decay=self.init_weight_decay
                )
                scheduler = CosineSchedule(optimizer=optimizer.base_optimizer,K=self.init_epoch)
                
                


            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            print(f'Debug later task self.optim: {self.optim}')
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            elif self.optim == 'sam-sgd':
                # optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                optimizer = SAM(
                    self._network.parameters(),
                    base_optimizer=optim.SGD,
                    rho=self.rho,
                    lr=self.lrate,
                    momentum=0.9,
                    weight_decay=self.weight_decay
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer.base_optimizer,T_max=self.epoch)

            elif self.optim == 'sam-adam':
                # optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                
                optimizer = SAM(
                    self._network.parameters(),
                    base_optimizer=optim.Adam,
                    rho=self.rho,
                    lr=self.lrate,
                    betas=(0.9,0.999),
                    weight_decay=self.weight_decay
                )
                scheduler = CosineSchedule(optimizer=optimizer.base_optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

            # 再次提取 cur_matrix，用于更新当前任务的 投影子空间矩阵（GPM 或 DualGPM）
            mat_list = []
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0

            # update_DualGPM() 用于多任务、双空间（key 与 value）信息的保留。
            # self.update_GPM(mat_list)
            self.update_DualGPM(mat_list)
            
            #这一步是为了后续任务使用 subspace 投影策略（retain/remove）
            # Projection Matrix Precomputation
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf=torch.Tensor(np.dot(self.feature_list[p],self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                self.feature_mat.append(Uf)

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            # self._network.eval()
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                # == 第一次 forward ==
                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

                if  'sam' in self.optim.lower():
                    # print('SAM')

                    # ✅ 启用 BN running stats：用于真实训练
                    enable_running_stats(self._network)

                    # == 第一次 backward ==
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    
                    # ❗禁用 BN running stats：避免污染 BN 统计
                    disable_running_stats(self._network)

                    # second forward-backward
                    logits = self._network(inputs)['logits']
                    second_loss = F.cross_entropy(logits, targets)

                    # == 第二次 backward ==
        
                    second_loss.backward()
                    optimizer.second_step(zero_grad=True)

                else:
                    # print('No-SAM')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    




                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets//self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_DualGPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-'*40)


    def update_GPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                self.feature_list.append(U[:,0:max(r,1)])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
            
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                    continue
                # update GPM
                Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    self.feature_list[i]=Ui
    
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            logging.info('Layer {} : {}/{}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-'*40)  
