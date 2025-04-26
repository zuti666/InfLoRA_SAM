# InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning

<div align="justify">
  This is the official implementation of our CVPR 2024 paper "Interference-Free Low-Rank Adaptation for Continual Learning".
In this paper, we propose a new parameter-efficient continual learning method called interference-free low-rank adaptation (InfLoRA). 
</div>

<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> 227a2fa (sam)
## Introduction

<div align="justify">
Continual learning requires the model to learn multiple tasks sequentially. In continual learning, the model should possess the ability to maintain its performance on old tasks (stability) and the ability to adapt to new tasks continuously (plasticity). Recently, parameter-efficient fine-tuning (PEFT), which involves freezing a pre-trained model and injecting a small number of learnable parameters to adapt to downstream tasks, has gained increasing popularity in continual learning. Although existing continual learning methods based on PEFT have demonstrated superior performance compared to those not based on PEFT, most of them do not consider how to eliminate the interference of the new task on the old tasks, which inhibits the model from making a good trade-off between stability and plasticity. In this work, we propose a new PEFT method, called interference-free low-rank adaptation (InfLoRA), for continual learning. InfLoRA injects a small number of parameters to reparameterize the pre-trained weights and shows that fine-tuning these injected parameters is equivalent to fine-tuning the pre-trained weights within a subspace. Furthermore, InfLoRA designs this subspace to eliminate the interference of the new task on the old tasks, making a good trade-off between stability and plasticity. Experimental results show that InfLoRA outperforms existing state-of-the-art continual learning methods on multiple datasets.
</div>

![InfLoRA.png](InfLoRA.png)

## Requisite

This code is implemented in PyTorch, and we perform the experiments under the following environment settings:

- python = 3.8
- torch = 1.10.0
- torchvision = 0.11.1
- timm = 0.6.7

<<<<<<< HEAD
## Install
```
conda env create -f environment.yaml
```
=======
>>>>>>> 227a2fa (sam)
I think the code can run under other versions of the environment, but I haven't tried.


## Dataset preparation
<<<<<<< HEAD
 * Create a folder `data/`
 * **CIFAR 100**: should automatically be downloaded
 * **ImageNet-R**: download dataset from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar. After unzipping, place it into `data/` folder 
 * **DomainNet**: download from http://ai.bu.edu/M3SDA/, place it into `data/` folder 

## Training
All commands should be run under the project root directory. Currently, the code has been validated on 1 A6000 GPU (48G) and 4 2080ti GPUs (11G).
=======
Please refer to the following links to download three standard class incremental learning benchmark datasets.

 * **CIFAR 100**: should automatically be downloaded
 * **ImageNet-R**: retrieve from: https://github.com/hendrycks/imagenet-r
 * **DomainNet**: retrieve from: http://ai.bu.edu/M3SDA/

## Training
All commands should be run under the project root directory. **The scripts are set up for 4 GPUs** but can be modified for your hardware.
>>>>>>> 227a2fa (sam)

### CIFAR100:
#### For InfLoRA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/cifar100_inflora.json 
=======
python main.py --device "1" --config configs/cifar100_inflora.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA-b5
```
<<<<<<< HEAD
python main.py --device your_device --config configs/cifar100_inflorab5.json 
=======
python main.py --device "1" --config configs/cifar100_inflorab5.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA+CA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/cifar100_infloraca.json 
=======
python main.py --device "1" --config configs/cifar100_infloraca.json 
>>>>>>> 227a2fa (sam)
```

### ImageNet-R (10 Tasks):
#### For InfLoRA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg10_inflora.json 
=======
python main.py --device "1" --config mimg10_inflora.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA-b5
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg10_inflorab5.json 
=======
python main.py --device "1" --config mimg10_inflorab5.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA+CA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg10_infloraca.json 
=======
python main.py --device "1" --config mimg10_infloraca.json 
>>>>>>> 227a2fa (sam)
```

### ImageNet-R (20 Tasks):
#### For InfLoRA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg20_inflora.json 
=======
python main.py --device "1" --config mimg20_inflora.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA-b5
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg20_inflorab5.json 
=======
python main.py --device "1" --config mimg20_inflorab5.json 
>>>>>>> 227a2fa (sam)
```

### ImageNet-R (5 Tasks):
#### For InfLoRA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg5_inflora.json 
=======
python main.py --device "1" --config mimg5_inflora.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA-b5
```
<<<<<<< HEAD
python main.py --device your_device --config configs/mimg5_inflorab5.json 
=======
python main.py --device "1" --config mimg5_inflorab5.json 
>>>>>>> 227a2fa (sam)
```

### DomainNet:
#### For InfLoRA
```
<<<<<<< HEAD
python main.py --device your_device --config configs/domainnet_inflora.json 
=======
python main.py --device "1" --config domainnet_inflora.json 
>>>>>>> 227a2fa (sam)
```

#### For InfLoRA-b5
```
<<<<<<< HEAD
python main.py --device your_device --config configs/domainnet_inflorab5.json 
```

## Citation

```bibtex
@inproceedings{liang2024inflora,
  title={InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning},
  author={Liang, Yan-Shuo and Li, Wu-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23638--23647},
  year={2024}
}
=======
python main.py --device "1" --config domainnet_inflorab5.json 
>>>>>>> 227a2fa (sam)
```


## Acknoledgements
We thank the following repos providing helpful components/functions in our work.

- [PyCIL](https://github.com/G-U-N/PyCIL)
- [S-Prompts](https://github.com/iamwangyabin/S-Prompts)


<<<<<<< HEAD
=======
>>>>>>> b84959c (fix bugs)
>>>>>>> 227a2fa (sam)

