import torch
import numpy as np
import time
import os
import os.path as osp
import argparse
import sys
import inspect
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.neural_process import MlpNeuralODEProcessAudio
from models.training import NeuralProcessTrainer
import joblib
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data_process import *

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--latent_only', action='store_true', help='Decode only latent state (no z)')
parser.add_argument('--exclude_time', action='store_true', help='Exclude time from the ODE')
parser.add_argument('--data', type=str, choices=['RotMNIST', 'VaryRotMNIST','COVID19'], default='COVID19')
parser.add_argument('--model', type=str, choices=['np', 'ndp'], default='ndp')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--load', type=eval, choices=[True, False], default=False)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_context', type=int, default=5, help='Context size')#maximum size is 5
parser.add_argument('--eval_num_context', type=int, default=5, help='Eval context size')
parser.add_argument('--num_extra_target', type=int, default=6,
    help='Maximum number of extra target points')
parser.add_argument('--r_dim', type=int, default=100, help='Dimension of the aggregated context')
parser.add_argument('--z_dim', type=int, default=50, help='Dim of the latent sampled variable')
parser.add_argument('--L_dim', type=int, default=25, help='Dimension of the latent ODE')
parser.add_argument('--h_dim', type=int, default=100, help='Dim of the hidden layers in the ODE')
parser.add_argument('--lr', type=float, default=1e-5, help="Model learning rate")
parser.add_argument('--use_y0', action='store_true', help="Whether to use initial y or not")
parser.add_argument('--min_save_epoch', type=int, default=5,
    help="Epoch from which to start saving the model.")
parser.add_argument('--use_all_targets', type=eval, choices=[True, False], default=True,
    help="Use all the points in the time-series as target.")

parser.add_argument('--win', type=int, default=5, help = 'number of points in each window')
parser.add_argument('--inputdim', type=int, default=385, help = 'input dimension')
parser.add_argument('--y_dim', type=int, default=1, help = 'output dimension')
parser.add_argument('--posweight', type=float, default=1.5, help = 'positive weight')
parser.add_argument('--is_aug', type=bool, default=True, help = 'adding augmentation')
parser.add_argument('--varname', type=str, default='audiorandcontextpos1.00001', help = 'variational name for test')
args = parser.parse_args()

# Set device.
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

def run():
    # Parse and print arguments.
    if args.exp_name is None:
        args.exp_name = str(time.time())
    print(args)

    # Make folder
    folder = osp.join('results', args.data, args.model)
    if not osp.exists(folder):
        os.makedirs(folder)

    # Create dataset
    data = joblib.load(filepath + 'Posdata_embed106.pk')
    fold = joblib.load(filepath + 'newposdata_fold106.pk')
    traindata, valdata, testdata, names = data_partiton_long(data, fold, lanset)
    data1 = joblib.load(filepath + 'Negdata_embed106.pk')  # 'Negdata_embed0618.pk'
    fold1 = joblib.load(filepath + 'newnegdata_fold106.pk')  # 'newnegdata_fold0617.pk'
    traindata1, valdata1, testdata1, names1 = data_partiton_long(data1, fold1, lanset)

    if args.is_aug:
        data_aug = joblib.load(filepath + 'Posdata_embed106_aug.pk')
        traindata_aug, valdata_aug, _, names = data_partiton_long(data_aug, fold, lanset)#samplecheck(trainlabelaug, trainlabelaug, trainlabelaug, 'pos')

        data_aug1 = joblib.load(filepath + 'Negdata_embed106_aug.pk')
        traindata_aug1, valdata_aug1, _, names = data_partiton_long(data_aug1, fold1, lanset)

    trinput = traindata['input']+traindata1['input'] \
              +traindata_aug['input']+traindata_aug1['input']
    trlabel = traindata['label']+traindata1['label']\
              +traindata_aug['label']+traindata_aug1['label']

    vlinput = valdata['input']+valdata1['input']\
              +valdata_aug['input']+valdata_aug1['input']
    vllabel = valdata['label']+valdata1['label']\
              +valdata_aug['label']+valdata_aug1['label']

    teinput = testdata['input']+testdata1['input']
    telabel = testdata['label']+testdata1['label']

    train_dataset = Dataclass(trinput,trlabel)
    val_dataset = Dataclass(vlinput, vllabel)
    test_dataset = Dataclass(teinput, telabel)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)

    initial_t = torch.tensor(0).view(1, 1, 1).to(device) #initial point should be updated

    neuralprocess = MlpNeuralODEProcessAudio(args.inputdim, args.y_dim, args.r_dim, args.z_dim,
                                        args.h_dim, args.L_dim, initial_t).to(device)
    if args.load:
        print('Loading model from existent path...')
        neuralprocess = torch.load(osp.join(folder, 'trained_model.pth')).to(device)

    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=args.lr)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
        num_context_range=(1, args.num_context), num_extra_target_range=(1, args.num_extra_target),
        use_y0=args.use_y0, use_all_targets=args.use_all_targets)

    logfile = open(savepath + 'covidtest' + str(args.lr) + '_log.txt', "w")
    logfile.write("INIT testing results:")
    logfile.write("\n")

    start_time = time.time()

    #best_nll = 1e15
    best_auc = 0.5
    tr_loss = []
    vl_loss = []
    vl_nll = []
    te_loss = []
    AUC = np.zeros((3,args.epochs,))
    TPR = np.zeros((3,args.epochs,))
    TNR = np.zeros((3,args.epochs,))
    TPR_TNR_9 = np.zeros((3,args.epochs,))

    for epoch in range(args.epochs):
        train_epoch_loss, x_context, y_context, x_target, y_target, p_y_pred = np_trainer.train_epoch(train_loader, args.posweight)
        val_mse, val_nll, x_cxt, y_cxt, x_tar, y_tar, y_val = np_trainer.eval_epoch(val_loader, context_size=1)
        print(f'Epoch: {epoch} | Train loss: {train_epoch_loss:.3f} | Validation MSE: {val_mse:.3f} '
              f'| Val NLL: {val_nll:.3f}')

        AUC[0,epoch], TPR[0,epoch], TNR[0,epoch], TPR_TNR_9[0,epoch] = get_metrics(p_y_pred, y_target, dis)
        AUC[1,epoch], TPR[1,epoch], TNR[1,epoch], TPR_TNR_9[1,epoch] = get_metrics(y_val, y_tar, dis)

        tr_loss.append(train_epoch_loss)
        vl_loss.append(val_mse)
        vl_nll.append(val_nll)

        logfile.write("Train Epoch {}/{}: AUC:{:.3f}, TPR:{:.3f}, TNR:{:.3f}, TPR_TNR_9:{:.3f}".format(
            epoch, args.epochs, AUC[0,epoch], TPR[0,epoch], TNR[0,epoch], TPR_TNR_9[0,epoch]))
        logfile.write("\n")
        logfile.write("Val Epoch {}/{}: AUC:{:.3f}, TPR:{:.3f}, TNR:{:.3f}, TPR_TNR_9:{:.3f}".format(
            epoch, args.epochs, AUC[1,epoch], TPR[1,epoch], TNR[1,epoch], TPR_TNR_9[1,epoch]))
        logfile.write("\n")

        if 0.5 * (TPR[1,epoch] + TNR[1,epoch]) > best_auc:# and epoch > args.min_save_epoch:
            print('New best validation obtained. Checkpointing model and plotting...')
            best_auc = 0.5 * (TPR[1,epoch] + TNR[1,epoch])
            torch.save(neuralprocess.state_dict(), osp.join(folder, args.varname + str(args.posweight) + str(args.lr) + 'trained_modelnew.pth'))

            test_mse, test_nll, x_cxtest, y_cxtest, \
            x_tartest, y_tartest, y_test = np_trainer.eval_epoch(test_loader, context_size=1)
            print(f'Epoch: {epoch} | Test MSE: {test_mse:.3f} '
                  f'| Test NLL: {test_nll:.3f}')

            AUC[2, epoch], TPR[2, epoch], TNR[2, epoch], TPR_TNR_9[2, epoch] = get_metrics(y_test,y_tartest, dis)
            te_loss.append(test_mse)

            logfile.write("Test Epoch {}/{}: AUC:{:.3f}, TPR:{:.3f}, TNR:{:.3f}, TPR_TNR_9:{:.3f}".format(
                epoch, args.epochs, AUC[2, epoch], TPR[2, epoch], TNR[2, epoch], TPR_TNR_9[2, epoch]))
            logfile.write("\n")

    end_time = time.time()
    print("==========================")
    print(f'Total time = {end_time - start_time}')
    print(f"Best MSE: {best_auc:.4f}")


if __name__ == "__main__":
    filepath = '/filepath/'  # model_vggtune/data/'
    savepath = '/savepath/'
    lanset = ['en', 'it', 'fr', 'de', 'es', 'ru', 'pt', 'NA'] # different lauguage of the data
    dis = 'Bernoulli'
    run()
