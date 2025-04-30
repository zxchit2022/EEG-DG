# coding:UTF-8
import torch
import numpy as np
import torch.nn as nn
import scipy.io as scio
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import random
import pywt
import mmd
import tSNE_4
from Shallow_Inception_Network_2source import DG_Network
import Dist_Loss
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.autograd as autograd
autograd.set_detect_anomaly(True)


class argparse():
    pass


args = argparse()
args.learning_rate = 0.0005
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.batch_size = 8
args.classes = 4
args.channels = 22
args.seed = 24
args.epochs = 500
args.momentum = 0.9
args.weight_decay = 0.05
args.save_path = "......"
#################################################
args.lammda = 0.1
args.beta = 0.1
args.gamma = 0.1
##################################################
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 创建数据集Dataset
class MyDataset_train(Dataset):
    def __init__(self, Feature, Label, Label_):
        self.Feature = Feature
        self.Label = Label
        self.Label_ = Label_

    def __getitem__(self, index):
        feature = torch.from_numpy(self.Feature[index])
        label = torch.tensor(self.Label[index])
        label_ = torch.tensor(self.Label_[index])
        return feature, label, label_

    def __len__(self):
        return len(self.Feature)


class MyDataset_test(Dataset):
    def __init__(self, Feature, Label):
        self.Feature = Feature
        self.Label = Label

    def __getitem__(self, index):
        feature = torch.from_numpy(self.Feature[index])
        label = torch.tensor(self.Label[index])
        return feature, label

    def __len__(self):
        return len(self.Feature)


def load_trainmat(i):
    path = "H:\Database\EEG\Motor Imagery\BCI Competition\BCI_IV\Data_Sets_2a\BCI_IV_2a_Subj{}_session1_2-6s.mat".format(i)
    data = scio.loadmat(path)

    Feature = data["Feature"].transpose(2, 1, 0)
    Label = data["Label"][:, 0]

    num = Feature.shape[0]
    num1 = int(num/2)

    # Source 1
    Feature1 = Feature[0:num1, :, :]
    Label1 = Label[0:num1]
    label1_ = np.zeros(Feature1.shape[0]) + 1
    source_domain1 = [Feature1, Label1, label1_]

    # Source 2
    Feature2 = Feature[num1:, :, :]
    Label2 = Label[num1:]
    label2_ = np.zeros(Feature2.shape[0]) + 2
    source_domain2 = [Feature2, Label2, label2_]

    return source_domain1, source_domain2


def load_testmat(j):
    path = "H:\Database\EEG\Motor Imagery\BCI Competition\BCI_IV\Data_Sets_2a\BCI_IV_2a_Subj{}_session2_2-6s.mat".format(j)
    data = scio.loadmat(path)
    Feature = data["Feature"].transpose(2, 1, 0)
    Label = data["Label"][:, 0]
    target_domain = [Feature, Label]

    return target_domain


def train(train_loader1, train_loader2, test_loader, model, optimizer):
    len_train_loader1 = len(train_loader1)
    len_train_loader2 = len(train_loader2)

    Train_Acc = []
    Domain_Acc = []
    Train_Loss = []
    CLC_Loss = []
    MMD_Loss = []
    MCD_Loss = []
    DOM_Loss = []
    Test_Acc = []
    Test_Loss = []
    Loss = []
    best_acc = 0

    for i in range(args.epochs):
        correct = 0
        correct_ = 0
        train_loss = []
        clc_loss = []
        dom_loss = []
        mmd_loss = []
        mcd_loss = []
        model.train()

        feat_train = []
        lab_train = []
        lab_domain = []

        iter_train1, iter_train2 = iter(train_loader1), iter(train_loader2)
        n_batch = min(len_train_loader1, len_train_loader2)


        for _ in range(n_batch):
            data_train1, label_train1, label_domain1 = next(iter_train1)
            data_train2, label_train2, label_domain2 = next(iter_train2)

            # for t_SNE
            label_train_ = np.concatenate((label_train1, label_train2), axis=0)
            label_domain_ = np.concatenate((label_domain1, label_domain2), axis=0)
            lab_train.extend(label_train_)
            lab_domain.extend(label_domain_)

            data_train1, label_train1 = data_train1.to(args.device), label_train1.to(args.device) - 1
            label_domain1 = label_domain1.to(args.device) - 1
            label_train1, label_domain1 = label_train1.long(), label_domain1.long()

            data_train2, label_train2 = data_train2.to(args.device), label_train2.to(args.device) - 1
            label_domain2 = label_domain2.to(args.device) - 1
            label_train2, label_domain2 = label_train2.long(), label_domain2.long()

            label_train = torch.cat((label_train1, label_train2), dim=0)
            label_domain = torch.cat((label_domain1, label_domain2), dim=0)

            out, dom, Feat_s, Feat = model(data_train1, data_train2)   # Feat for reducing the mmd loss  , Feat , alpha
            feat_train.extend(Feat)

            # classification loss and domian loss
            clc_loss_ = loss_clc(out, label_train)
            clc_loss.append(clc_loss_.item())
            dom_loss_ = loss_domain(dom, label_domain)
            dom_loss.append(dom_loss_.item())

            # predict labels
            label_pred = torch.max(out, 1)[1]
            correct += (label_pred == label_train).sum().item()
            dom_pred = torch.max(dom, 1)[1]
            correct_ += (dom_pred == label_domain).sum().item()

            # domain adaptation for invariant feature learning
            Feat_s_ = torch.mean(torch.stack((Feat_s[0], Feat_s[1]), dim=0), dim=0)
            mmd_loss1 = loss_mmd(Feat_s[0], Feat_s_)
            mmd_loss2 = loss_mmd(Feat_s[1], Feat_s_)
            mmd_loss_ = (mmd_loss1 + mmd_loss2) / 2
            mmd_loss.append(mmd_loss_.item())

            # conditional distribution loss
            mcd_loss_ = loss_mcd(Feat_s[0], label_train, Feat_s[1], label_train, 0.1)
            mcd_loss.append(mcd_loss_.item())

            # sum of losses
            loss = clc_loss_ + args.lammda * mmd_loss_ + args.gamma * mcd_loss_ + args.beta * dom_loss_
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # scheduler.step()

        train_acc = (correct / (len(train_loader1.dataset) + len(train_loader2.dataset))) * 100
        domain_acc = (correct_ / (len(train_loader1.dataset) + len(train_loader2.dataset))) * 100
        test_acc, test_loss, feat_test, lab_test = test(model, test_loader)

        Train_Loss.append(np.average(train_loss))
        CLC_Loss.append(np.average(clc_loss))
        MMD_Loss.append(np.average(mmd_loss))
        MCD_Loss.append(np.average(mcd_loss))
        DOM_Loss.append(np.average(dom_loss))
        # MTC_Loss.append(np.average(mtc_loss))
        Train_Acc.append(train_acc)
        Domain_Acc.append(domain_acc)
        Test_Loss.append(test_loss)
        Test_Acc.append(test_acc)

        Loss = [CLC_Loss, MMD_Loss, MCD_Loss, DOM_Loss]   #, MTC_Loss

        if best_acc < test_acc:
            best_acc = test_acc
            feat_tsne = [feat_train, lab_train, lab_domain, feat_test, lab_test]

        if i % 10 == 0:
            print("Epoch {}, Train loss:{:.2f}, Train acc:{:.2f}, Test loss:{:.2f}, Test acc:{:.2f}".format(i,
                  np.mean(train_loss), train_acc, test_loss, test_acc))

    return Train_Loss, Loss, Train_Acc, Domain_Acc, Test_Loss, Test_Acc, feat_tsne, best_acc     #


def test(model, test_loader):
    model.eval()
    test_loss = []
    correct = 0

    feat_test = []
    lab_test = []

    with torch.no_grad():
        for data_test, label_test in test_loader:
            lab_test.extend(label_test)

            data_test, label_test = data_test.to(args.device), label_test.to(args.device) - 1
            label_test = label_test.long()
            output, _, feat_test_ = model.predict(data_test)  # , alpha
            feat_test.extend(feat_test_)

            # print(output.shape)
            loss = loss_clc(output, label_test)
            test_loss.append(loss.item())

            label_pred = torch.max(output, 1)[1]
            correct += (label_pred == label_test).sum().item()

    acc = (correct / len(test_loader.dataset)) * 100

    return acc, np.mean(test_loss), feat_test, lab_test


if __name__ == '__main__':
    print("Parameters: epochs{}, learning_rate{}, weight_decay{}, batch_size{}".format(
        args.epochs, args.learning_rate, args.weight_decay, args.batch_size))
    print("mmd_loss_{}, dom_loss_{}, mcd_loss_{}".format(args.lammda, args.beta, args.gamma))
    Ave_Acc = []

    for sub in range(1, 10):

        # load data
        source_domain1, source_domain2 = load_trainmat(sub)
        target_domain = load_testmat(sub)
        print("############################################")
        print("Source1:", source_domain1[0].shape, source_domain1[1].shape, source_domain1[2].shape)
        print("Source2:", source_domain2[0].shape, source_domain2[1].shape, source_domain2[2].shape)
        print("Target:", target_domain[0].shape, target_domain[1].shape)
        print("############################################")

        # (trials, 1, channels, sampling_points)
        train_data1 = np.expand_dims(source_domain1[0], axis=1)
        train_data2 = np.expand_dims(source_domain2[0], axis=1)
        test_data = np.expand_dims(target_domain[0], axis=1)

        # standardize
        train_mean = np.mean(np.concatenate((train_data1, train_data2), axis=0))
        train_std = np.std(np.concatenate((train_data1, train_data2), axis=0))
        train_data1 = (train_data1 - train_mean) / train_std
        train_data2 = (train_data2 - train_mean) / train_std
        test_data = (test_data - train_mean) / train_std

        # tensor
        train_data1_ = MyDataset_train(train_data1, source_domain1[1], source_domain1[2])
        train_data2_ = MyDataset_train(train_data2, source_domain2[1], source_domain2[2])
        test_data_ = MyDataset_test(test_data, target_domain[1])

        train_loader1 = DataLoader(train_data1_, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loader2 = DataLoader(train_data2_, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data_, batch_size=16, shuffle=True, drop_last=True)

        my_model = DG_Network(args.classes, args.channels).to(args.device)  # 22 channels, 500 hidden_dim, 100 out_dim, 4 classes, 3 domains
        # print(my_model)

        loss_clc = nn.CrossEntropyLoss()
        loss_domain = nn.CrossEntropyLoss()
        loss_mmd = mmd.MMD_loss()
        loss_mcd = Dist_Loss.Dist_Loss()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95)

        Train_Loss, Loss, Train_Acc, Domain_Acc, Test_Loss, Test_Acc, feat_tsne, best_acc = train(train_loader1, train_loader2, test_loader, my_model, optimizer)  #
        print("Subject {}, best Acc: {:.2f}%".format(sub, best_acc))

        Ave_Acc.append(best_acc)



