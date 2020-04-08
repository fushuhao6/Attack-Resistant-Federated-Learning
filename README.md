# Attack-Resistant Federated Learning with Residual-based Reweighting


This repository is implemented by [Shuhao Fu](https://github.com/howardmumu) and [Chulin Xie](https://github.com/AlphaPav).

## Introduction


This is a PyTorch implementation of our [paper](https://arxiv.org/abs/1912.11464). We present a novel aggregation algorithm with residual-based reweighting to defend federated learning. Our aggregation algorithm combines repeated median regression with the reweighting scheme in iteratively reweighted least squares. Our experiments show that our aggregation algorithm outperforms other alternative algorithms in the presence of label-flipping, backdoor, and Gaussian noise attacks. We also provide theoretical guarantees for our aggregation algorithm.
  * This repository used code from [federated learning](https://github.com/shaoxiongji/federated-learning).

## Citing Attack-Resistant Federated Learning
If you find Attack-Resistant Federated Learning useful in your research, please consider citing:
```
@article{fu2019attackresistant,
    title={Attack-Resistant Federated Learning with Residual-based Reweighting},
    author={Shuhao Fu and Chulin Xie and Bo Li and Qifeng Chen},
    journal={arXiv:1912.11464},
    year={2019}
}
```


## Main Results
Our algorithm can successfully defend Gaussian Noise Attacks, Label-Flipping Attacks and Backdoor Attacks. 

| # of attackers  | 0      | 1      | 2      | 3      | 4      | Average |
|-----------------|--------|--------|--------|--------|--------|---------|
| FedAvg          | 88.96% | 85.74% | 82.49% | 82.35% | 82.11% | 84.33%  |
| Median          | 88.11% | 87.69% | 87.15% | 85.85% | 82.01% | 86.16%  |
| Trimmed Mean    | 88.70% | 88.52% | 87.44% | 85.36% | 82.35% | 86.47%  |
| Repeated Median | 88.60% | 87.76% | 86.97% | 85.77% | 81.82% | 86.19%  |
| FoolsGold       | 9.70%  | 9.57%  | 10.72% | 11.42% | 9.98%  | 10.28%  |
| Ours            | 89.17% | 88.60% | 86.66% | 86.09% | 85.81% | 87.27%  |

*Results of label-flipping attacks on CIFAR-10 dataset with different numbers of attackers.*

![MNIST](images/MNIST.png?raw=true) ![backdoor](images/backdoor.png?raw=true)

*Results of label-flipping attacks on the MNIST dataset (left). Result of backdoor attack success rate
on CIFAR-10 (right).*


## Requirements: Software

1. Pytorch from [the offical repository](https://pytorch.org/).
2. Install tensorboardX.
```
pip install tensorboardX
```


## Preparation for Training & Testing
1. The code will automatically download MNIST dataset.

2. Loan dataset preprocess:
Download Lending Club Loan Data dataset from https://www.kaggle.com/wendykan/lending-club-loan-data into the dir `FedAvg/loan`

```
cd FedAvg/loan
sh process_loan_data.sh
```


## Usage
### Label-flipping attack experiments
Label Flipping attack on MNIST
```
python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.1 --num_attackers 4 --attacker_ep 10 --num_users 100 --attack_label 1 --agg irls
```

Label Flipping attack on CIFAR-10
```
python main_nn.py --model resnet --dataset cifar --epochs 100 --gpu 0 --iid 0 --local_bs 64 --num_users 6 --num_attackers 4 --attacker_ep 10 --attack_label 3 --irls
```

Change `--agg` tag to select aggregation algorithm and change `--num_attackers` to specify the number of attackers. Note that in CIFAR experiments the `--num_users` + `--num_attackers` should equal 10.

### Backdoor attack experiments

MNIST naive approach (multiple-shot)

```
python main_nn.py --epochs 200 --local_bs 64 --num_attackers 1 --attacker_ep 10 --num_users 9 --is_backdoor true
```

MNIST model replacement (single-shot)

```
python main_nn.py --epochs 200 --local_bs 64 --num_attackers 1 --attacker_ep 10 --num_users 9 --is_backdoor true --backdoor_scale_factor 10 --backdoor_single_shot_scale_epoch 5
```

Loan

```
python main_nn.py --model loannet --dataset loan --epochs 100 --local_bs 64 --num_attackers 1 --local_ep 5 --attacker_ep 10 --num_users 9 --is_backdoor true --backdoor_label 1 --backdoor_per_batch 10 --lr 0.01 
```

CIFAR-10

```
python main_nn.py --model resnet --dataset cifar --num_channels 3 --epochs 200 --local_bs 64 --num_attackers 1 --attacker_ep 10 --num_users 9 --is_backdoor true
```
