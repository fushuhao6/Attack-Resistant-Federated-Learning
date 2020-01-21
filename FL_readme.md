### Loan dataset preprocess

download Lending Club Loan Data dataset from https://www.kaggle.com/wendykan/lending-club-loan-data into the dir `loan`

```
cd loan
sh process_loan_data.sh
```

###backdoor experiment

Mnist naive approach (multiple-shot)

```
python main_nn.py --epochs 200 --local_bs 64 --num_attackers 1 --attacker_ep 10 --num_users 9 --is_backdoor true
```

Mnist model replacement (single-shot)

```
python main_nn.py --epochs 200 --local_bs 64 --num_attackers 1 --attacker_ep 10 --num_users 9 --is_backdoor true --backdoor_scale_factor 10 --backdoor_single_shot_scale_epoch 5
```

Loan

```
python main_nn.py --model loannet --dataset loan --epochs 100 --local_bs 64 --num_attackers 1 --local_ep 5 --attacker_ep 10 --num_users 9 --is_backdoor true --backdoor_label 1 --backdoor_per_batch 10 --lr 0.01 
```

Cifar 

```
python main_nn.py --model resnet --dataset cifar --num_channels 3 --epochs 200 --local_bs 64 --num_attackers 1 --attacker_ep 10 --num_users 9 --is_backdoor true
```