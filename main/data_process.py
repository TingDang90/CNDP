import torch
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import joblib
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import metrics

def lan_check(uids):
    '''This function checks and balances the language of the users'''
    user_all = joblib.load('metadata_posusers.pk') # this loads the meta data for positive and negative users
    user_all1 = joblib.load('metadata_negusers.pk')
    user_all.update(user_all1)

    lan = []
    lan1 = {}
    for uid in uids:
        if 'locale' in user_all[uid].keys() and len(user_all[uid]['locale'])>0:
            lanorig = user_all[uid]['locale']
            if lanorig[0] != 'en':  # LOCALE
                if len(uid) == 12: # IOS system
                    if lanorig[0] not in ['en', 'it']:
                        lanorig = ['en']
                if len(uid) == 10: # Android system
                    if lanorig[0] not in ['en', 'it', 'es', 'de', 'pt', 'el', 'fr', 'ru', 'ro', 'zh', 'hi']:
                        lanorig = ['en']

            lan1[uid] = lanorig
            lan.append(lanorig)
        else:
            lan1[uid] = ['NA']
            lan.append(['NA'])

    ind = [index for index, value in enumerate([len(i) for i in lan]) if value == 0]
    for i in ind[::-1]:
        del lan[i]
    lan = [i[0] for i in lan]
    return lan1, lan

def datalong(data,sfold, lan, lanset):
    s_data = {}
    s_data['input'] = []
    s_data['tsp'] = []
    s_data['label'] = []
    s_data['lan'] = []
    names = []

    cnt = 0

    for uid in sfold:
        print(uid)
        cnt += len(data[uid])
        ########## remove the first two points are more than 14 days back
        if uid in ['ZJ0PwoVS8UYE','1VRgDTzgPofR']:
            data[uid] = dict([(key-2, value) for key,value in data[uid].items() if key > 1])
            ref = data[uid][0]['time']
            for k in data[uid]:
                data[uid][k]['time'] = data[uid][k]['time'] - ref
        if uid == 'kEaqI3uZIz':
            data[uid] = dict([(key-15, value) for key,value in data[uid].items() if key > 14])
            ref = data[uid][0]['time']
            for k in data[uid]:
                data[uid][k]['time'] = data[uid][k]['time'] - ref

        lanvec = np.zeros((len(lanset),))
        lanvec[lanset.index(lan[uid][0])] = 1

        tt  = np.array([data[uid][j]['time'] for j in range(len(data[uid]))])
        tt= tt - tt[0]
        lab = np.array([data[uid][j]['label'] for j in range(len(data[uid]))])

        tt = np.array([tt[kk] for kk in np.where(lab[:, 0] != 0.5)[0]])
        tt = tt - tt[0]
        lab = np.vstack([lab[kk] for kk in np.where(lab[:, 0] != 0.5)[0]])

        for k in [j for j in data[uid][0].keys() if 'embed' in j]:
            fea = [data[uid][j][k] for j in range(len(data[uid]))]
            temp = np.vstack([fea[kk] for kk in np.where(lab[:,0] != 0.5)[0]])
            temp = np.concatenate((temp, np.expand_dims(tt.astype('float32'), -1)), -1)

            s_data['input'] = s_data['input'] + [torch.from_numpy(temp).float().to(device)]
            s_data['tsp'] = s_data['tsp'] + [torch.from_numpy(tt).float().to(device)]
            s_data['label'] = s_data['label'] + [torch.from_numpy(np.vstack(lab)).float().to(device)]  # [seq_lab[kk]]
            s_data['lan'] = s_data['lan'] + [torch.from_numpy(lanvec).float().to(device)]
    return s_data, names

def data_partiton_long(data,sfold,lanset):
    sdata = {}
    sdata['train']={}
    sdata['val'] = {}
    sdata['test'] = {}
    for pname in ['train','val','test']:
        landic, _ = lan_check(sfold[pname])
        ###################### interpolate the features ###################### ###################### ######################
        sdata[pname], names= datalong(data,sfold[pname], landic, lanset)
    return sdata['train'],sdata['val'], sdata['test'], names

def get_metrics(probs, labels, dis):
    if dis == 'Gaussian':
        probs = np.concatenate([i.loc[0,1:,0].detach().cpu().numpy() for i in probs])
        probs = np.squeeze(probs)
        label = np.vstack([i[0][1:].cpu().detach().numpy() for i in labels])
        label = np.squeeze(label)
    if dis == 'Bernoulli':
        probs = np.concatenate([i.probs[0,1:,0].detach().cpu().numpy() for i in probs])
        probs = np.squeeze(probs)
        label = np.vstack([i[0][1:].cpu().detach().numpy() for i in labels])
        label = np.squeeze(label)
    predicted = []
    for i in range(len(probs)):
        if probs[i] > 0.5:
            predicted.append(1)
        else:
            predicted.append(0)

    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)

    label = [i for i in label if i != 0.5]
    predicted = [j for i,j in zip(label,predicted) if i != 0.5]
    probs = [j for i,j in zip(label,probs) if i != 0.5]

    auc = metrics.roc_auc_score(label, probs)
    precision, recall, _ = metrics.precision_recall_curve(label, probs)

    TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    fpr, tpr, thresholds = metrics.roc_curve(label, probs)
    index = np.where(tpr > 0.9)[0][0] - 1
    print('AUC:' + "{:.2f}".format(auc) +
          ' Sensitivity:' + "{:.2f}".format(TPR) +
          ' Specificity:' + "{:.2f}".format(TNR) +
          ' spe@90%sen:' + "{:.2f}".format(1 - fpr[index]))

    return auc, TPR, TNR, 1 - fpr[index]

class Dataclass(Dataset):
    def __init__(self, x, y):
        self.num_samples = len(x)
        self.x_dim = args.inputdim  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        for i in range(len(x)):
            self.data.append((x[i], y[i][:,1:]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)