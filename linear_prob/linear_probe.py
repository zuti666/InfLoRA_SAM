import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import tensor2numpy, accuracy
from copy import deepcopy
import argparse
import json
from datetime import datetime
def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def linear_probe(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args['device'] = [0]
    args['net_type'] = 'sip'


    # 添加保存结果路径

    # 创建模型
    model = factory.get_model(args['model_name'], args)
    
    # 加载主干模型（训练好的 ViT encoder）
    backbone_ckpt = args['pretrained_ckpt']  # 传入预训练模型路径
    print(f"Loading pretrained encoder from: {backbone_ckpt}")
    state_dict = torch.load(backbone_ckpt, map_location='cpu')

    for k in list(state_dict.keys()):
        if 'classifier_pool' in k:  # 删除旧分类头
            del state_dict[k]
    model._network.load_state_dict(state_dict, strict=False)

    # 替换新的线性分类层
    model._network.classifier_pool = nn.ModuleList([
        nn.Linear(args["embd_dim"], args["probe_classes"], bias=True)
    ])
    model._network.numtask = 1  # 仅一个任务

    # 冻结所有参数，仅训练线性头
    for param in model._network.parameters():
        param.requires_grad = False
    for param in model._network.classifier_pool[0].parameters():
        param.requires_grad = True

    model._network.to(device)

    # 加载新任务数据集
    probe_data = DataManager(args['probe_dataset'], shuffle=True, seed=0,
                             init_cls=args["probe_classes"], increment=0, args=args)
    train_set = probe_data.get_dataset(np.arange(0, args["probe_classes"]), source='train', mode='train')
    test_set = probe_data.get_dataset(np.arange(0, args["probe_classes"]), source='test', mode='test')

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model._network.parameters()), lr=1e-3)
    epochs = args.get("probe_epochs", 20)

    # 训练线性分类器
    for epoch in range(epochs):
        model._network.train()
        total_loss, correct, total = 0., 0, 0
        for _, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model._network(inputs)['logits']
            loss = F.cross_entropy(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += len(targets)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.3f}, Acc: {acc:.2f}%")

    # 测试性能
    model._network.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model._network(inputs)['logits']
            preds = outputs.argmax(dim=1)
            y_pred.append(preds.cpu())
            y_true.append(targets.cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    test_acc = (y_pred == y_true).float().mean().item() * 100
    print(f"Final Linear Probe Test Accuracy: {test_acc:.2f}%")

def multi_task_linear_probe(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args['device'] = [0]
    args['net_type'] = 'sip'
    args['reload_model'] = True

    # === 创建输出和统一日志路径 ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("logs_probe", args['probe_dataset'], f"probe_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "main.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"===> [Init] Start Multi-Task Linear Probe: {args['probe_task_ids']}")

    results = {}

    for task_id in args['probe_task_ids']:
        logging.info(f"\n===== Linear-Probe for Task {task_id} =====")

        # === 构建模型并加载主干权重 ===
        model = factory.get_model(args['model_name'], args)
        state_dict = torch.load(args['pretrained_ckpt'], map_location='cpu')
        for k in list(state_dict.keys()):
            if 'classifier_pool' in k:
                del state_dict[k]
        missing_keys, unexpected_keys = model._network.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded pretrained model: {args['pretrained_ckpt']}")
        logging.info(f"Missing keys: {missing_keys}")
        logging.info(f"Unexpected keys: {unexpected_keys}")

        # === 替换线性头并冻结主干 ===
        model._network.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], args["probe_classes"], bias=True)
        ])
        model._network.numtask = 1
        for name, param in model._network.named_parameters():
            param.requires_grad = ('classifier_pool' in name)
        model._network.to(device)

        # === 加载数据集 ===
        probe_data = DataManager(args['probe_dataset'], shuffle=True, seed=task_id,
                                 init_cls=args["probe_classes"], increment=0, args=args)
        train_set = probe_data.get_dataset(np.arange(0, args["probe_classes"]), source='train', mode='train')
        test_set = probe_data.get_dataset(np.arange(0, args["probe_classes"]), source='test', mode='test')

        train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
        test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model._network.parameters()), lr=1e-3)
        epochs = args.get("probe_epochs", 20)

        logging.info(f"===> Start training task {task_id} for {epochs} epochs")
        for epoch in range(epochs):
            model._network.train()
            total_loss, correct, total = 0., 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model._network(inputs)['logits']
                loss = F.cross_entropy(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total += len(targets)

            acc = 100 * correct / total
            logging.info(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.3f}, Acc: {acc:.2f}%")

        # === 测试 ===
        model._network.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model._network(inputs)['logits']
                preds = outputs.argmax(dim=1)
                y_pred.append(preds.cpu())
                y_true.append(targets.cpu())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        test_acc = (y_pred == y_true).float().mean().item() * 100
        logging.info(f"===> [Eval] Task {task_id} - Linear Probe Accuracy: {test_acc:.2f}%")

        # === 保存线性头参数和准确率结果 ===
        head_save_path = os.path.join(output_dir, f"linear_head_task{task_id}.pth")
        torch.save(model._network.classifier_pool[0].state_dict(), head_save_path)

        results[f"task_{task_id}"] = {
            "test_accuracy": round(test_acc, 2),
            "head_ckpt": head_save_path
        }

    # === 保存结果 JSON ===
    results_path = os.path.join(output_dir, "probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"\n✅ All results saved to: {results_path}")





def main():
    parser = argparse.ArgumentParser(description="Linear Probe on Pretrained InfLoRA")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file.')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'], help='Probe mode')
    args = parser.parse_args()

    config = load_json(args.config)
    if args.mode == 'single':
        linear_probe(config)
    else:
        multi_task_linear_probe(config)


if __name__ == "__main__":
    main()

