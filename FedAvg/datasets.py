import numpy as np
from torchvision import datasets, transforms
import random
import copy

from loan.LoanHelper import LoanHelper, LoanDataset


def build_datasets(args):
    # load dataset and split users
    attackers = None
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))
        if args.is_backdoor:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users + args.num_attackers)
            else:  # dirichlet sample users
                dict_users = sample_dirichlet_train_data(dataset_train, args.num_users, args.num_attackers)
        elif args.iid:
            dict_users = iid(dataset_train, args.num_users + args.num_attackers)
        else:
            # sample users
            if args.single:
                dict_users = mnist_noniid_with_sybil(dataset_train, args.num_users, args.num_attackers)
            elif args.fix_total:
                assert args.iid == False
                # valid seed 3 4 9 13 1237
                dict_users, user_has_one, user_has_zero = mnist_noniid_fixed_total(dataset_train, args.num_users)
                np.random.seed(args.seed)
                if args.attack_label < 0 and args.donth_attack:
                    attackers = np.random.choice(user_has_zero, args.num_attackers, replace=False)
                    print('attackers {} are selected from {}'.format(attackers, user_has_zero))
                else:
                    attackers = np.random.choice(user_has_one, args.num_attackers, replace=False)
                    print('attackers {} are selected from {}'.format(attackers, user_has_one))
            else:
                dict_users = mnist_refined_with_sybil(dataset_train, args.num_users, args.num_attackers)
    elif args.dataset == 'cifar':
        print('==> Preparing CIFAR data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform_train, download=True)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform_test, download=True)
        if args.iid:
            dict_users = iid(dataset_train, args.num_users + args.num_attackers)
        else:  # dirichlet sample users
            dict_users = sample_dirichlet_train_data(dataset_train, args.num_users, args.num_attackers)
    elif args.dataset == 'loan':
        print('==> Preparing lending-club-loan-data..')
        filepath = './data/lending-club-loan-data/loan_processed.csv'
        loanHelper = LoanHelper(filepath)
        dataset_train = loanHelper.dataset_train
        dataset_test = loanHelper.dataset_test
        if args.iid:
            dict_users = iid(dataset_train, args.num_users + args.num_attackers)
        else:  # non-iid sample by states
            dict_users = loan_sample_by_state(loanHelper, args.num_users + args.num_attackers)
    else:
        exit('Error: unrecognized dataset')
    if args.dataset == 'loan':
        test_users = test_sampling_as_numbers_bylabels(loanHelper.test_labels, 9)
    else:
        test_users = test_sampling_as_numbers(dataset_name=args.dataset, dataset=dataset_test, num_labels=10)

    return dataset_train, dataset_test, dict_users, test_users, attackers


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        print('assigned {} data to user {}'.format(len(dict_users[i]), i))
    return dict_users


def mnist_noniid_with_sybil(dataset, num_users, num_attackers):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([]) for i in range(num_users + num_attackers)}
    labels = dataset.train_labels.numpy()
    idxs = np.arange(len(labels))
    u = np.unique(labels)

    idxs_labels = np.vstack((idxs, labels))
    for i in range(num_users):
        idx = i % 10
        idx_labels = [idxs_labels[0][l] for l in range(len(labels)) if idxs_labels[1][l] == idx]
        assert len(idxs_labels) > 0
        print('assigned label {} \"{}\" to user {}'.format(len(idx_labels), i, i))
        dict_users[i] = np.concatenate((dict_users[i], idx_labels), axis=0)
    for j in range(num_users, num_users + num_attackers):
        idx_labels = [idxs_labels[0][l] for l in range(len(labels)) if idxs_labels[1][l] == 1]
        dict_users[j] = np.concatenate((dict_users[j], idx_labels), axis=0)
        print('assigned label {} \"{}\" to attacker {}'.format(len(idx_labels), 1, j))
    return dict_users


def mnist_noniid_fixed_total(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users: total number of users, including attackers
    :return:
    """
    assert num_users == 100, 'currently only support 100 users'
    num_shards = 2 * num_users
    num_imgs = int(60000 / num_shards)
    assert num_shards % 10 == 0
    num2ids = {i: np.array(range(int(num_shards / 10))) + int(num_shards / 10) * i for i in range(10)}
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]                     # (0 | 1 | 2 | ... | 9)

    # divide and assign
    user_has_one = []
    user_has_zero = []
    for i in range(num_users):
        if i == 23:
            print(23)
        replace = False
        # if len(list(num2ids.keys())) < 2:
        #     replace = True
        rand_cls = np.random.choice(list(num2ids.keys()), 2, replace=replace)
        if 1 in rand_cls:
            user_has_one.append(i)
        if 0 in rand_cls:
            user_has_zero.append(i)
        # print(num2ids)
        rand_set = []
        for cls_id in rand_cls:
            assert len(num2ids[cls_id]) > 0
            rand_id = np.random.choice(num2ids[cls_id], 1)
            rand_set.append(rand_id[0])
            num2ids[cls_id] = list(set(num2ids[cls_id]) - set(rand_id))
            if len(num2ids[cls_id]) == 0:
                num2ids.pop(cls_id)

        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users, user_has_one, user_has_zero


def mnist_refined_with_sybil(dataset, num_users, num_attackers):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([]) for i in range(num_users + num_attackers)}
    labels = np.array(dataset.train_labels)
    idxs = np.arange(len(labels))

    idxs_labels = np.vstack((idxs, labels))
    for i in range(num_users):
        idx = [i % 10, (i + 5 + int(i / 5)) % 10 ] #np.random.randint(9) + 1) % 10]
        idx_labels = [idxs_labels[0][l] for l in range(len(labels)) if idxs_labels[1][l] in idx]
        assert len(idxs_labels) > 0
        dict_users[i] = np.concatenate((dict_users[i], idx_labels), axis=0)
        print('assigned label {} \"{}\" and \"{}\" to user {}'.format(len(idx_labels), idx[0], idx[1], i))

    for j in range(num_users, num_users + num_attackers):
        idx = [1, 2 + j - num_users]#(1 + np.random.randint(9) + 1) % 10]
        idx_labels = [idxs_labels[0][l] for l in range(len(labels)) if idxs_labels[1][l] in idx]
        dict_users[j] = np.concatenate((dict_users[j], idx_labels), axis=0)
        print('assigned label {} \"{}\" and \"{}\" to attacker {}'.format(len(idx_labels), idx[0], idx[1], j))
    return dict_users


def loan_sample_by_state(loanHelper,num_users):
    keys=copy.deepcopy(loanHelper.state_keys)
    random.shuffle(keys)
    print("all addr_state :", keys)
    dict_users = {i: np.array([]) for i in range(num_users)}
    for i in range(num_users):
        idxs= loanHelper.dict_by_states[keys[i]]
        dict_users[i]=np.array(idxs)
        print(f'assigned {len(idxs)} data in addr_state {keys[i]} to user {i}')
    return dict_users


def sample_dirichlet_train_data(dataset, num_users, num_attackers, alpha=0.9):
    classes = {}
    for idx, x in enumerate(dataset):
        _, label = x
        # label=label.item() # for gpu
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())
    class_size = len(classes[0])
    num_participants= num_users + num_attackers
    dict_users = {i: np.array([]) for i in range(num_users + num_attackers)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_participants * [alpha]))
        for user in range(num_participants):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            dict_users[user] = np.concatenate((dict_users[user], np.array(sampled_list)), axis=0)
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]

    # shuffle data
    for user in range(num_participants):
        random.shuffle(dict_users[user])
    return dict_users


def test_sampling_as_numbers(dataset_name, dataset, num_labels):
    dict_users = {i: np.array([]) for i in range(num_labels)}
    if dataset_name == 'mnist':
        labels = np.array(dataset.test_labels)
    elif dataset_name == 'cifar':
        labels = np.array(dataset.test_labels)
    else:
        print('dataset not supported')
        exit(-1)
    idxs = np.arange(len(labels))

    idxs_labels = np.vstack((idxs, labels))
    for i in range(num_labels):
        idx_labels = [idxs_labels[0][l] for l in range(len(labels)) if idxs_labels[1][l] == i]
        assert len(idxs_labels) > 0
        dict_users[i] = np.concatenate((dict_users[i], idx_labels), axis=0)
    return dict_users


def test_sampling_as_numbers_bylabels(test_labels, num_labels):
    dict_users = {i: np.array([]) for i in range(num_labels)}
    labels = np.array(test_labels)
    idxs = np.arange(len(labels))

    idxs_labels = np.vstack((idxs, labels))
    for i in range(num_labels):
        idx_labels = [idxs_labels[0][l] for l in range(len(labels)) if idxs_labels[1][l] == i]
        assert len(idxs_labels) > 0
        dict_users[i] = np.concatenate((dict_users[i], idx_labels), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    # valid seed 3 4 9 13 1237 ...
    np.random.seed(1237)
    d = mnist_noniid_fixed_total(dataset_train, num)
