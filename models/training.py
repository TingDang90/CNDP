import torch

from tqdm import tqdm
from typing import Tuple
from random import randint
from torch.distributions.kl import kl_divergence
from torch.distributions import Bernoulli, Normal
from models.utils import context_target_split as cts
from models.neural_process import TimeNeuralProcess
from torch.utils.data import DataLoader
import numpy as np
import scipy
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

class TimeNeuralProcessTrainer:
    """
    Class to handle training of Neural Processes.
    Code adapted from https://github.com/EmilienDupont/neural-processes

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.TimeNeuralProcess, neural_process.NeuralODEProcess instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    """
    def __init__(self,
                 device: torch.device,
                 neural_process: TimeNeuralProcess,
                 optimizer: torch.optim.Optimizer,
                 num_context_range: Tuple[int, int],
                 num_extra_target_range: Tuple[int, int],
                 max_context=None,
                 use_all_targets=False,
                 use_y0=True):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.max_context = max_context
        self.use_all_targets = use_all_targets
        self.use_y0 = use_y0

        self.epoch_loss_history = []
        self.epoch_nfe_history = []
        self.epoch_mse_history = []
        self.epoch_logp_history = []

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs: int):
        """
        Trains Neural (ODE) Process.

        Parameters
        ----------
        train_data_loader : Data loader to use for training
        val_data_loader: Data loader to use for validation
        epochs: Number of epochs to train for
        """
        self.neural_process.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            epoch_loss = self.train_epoch(train_data_loader)
            # some bit about the self.epoch_nfe_history.append if we want to track nfe in training
            self.epoch_loss_history.append(epoch_loss)
            self.eval_epoch(val_data_loader)

    def train_epoch(self, data_loader):
        epoch_loss = 0.
        self.neural_process.train()
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer.zero_grad()

            # Extract data
            x, y = data
            points = x.size(1)

            # Sample number of context and target points
            num_context = randint(*self.num_context_range)
            num_extra_target = randint(*self.num_extra_target_range)
            if self.use_all_targets:
                num_extra_target = points - num_context

            # Create context and target points and apply neural process
            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))#x_context are the points of interpolation; and x_target are the points including both interpolation and extrapolation
            y0 = y0.to(self.device)#y0 is the first point of the sequence if given
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)

            p_y_pred, q_target, q_context = (
                self.neural_process(x_context, y_context, x_target, y_target, y0))
            loss = self._loss(p_y_pred, y_target, q_target, q_context)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.cpu().item()

        return epoch_loss / len(data_loader), x_context, y_context, x_target, y_target, p_y_pred #only output the last batch

    def eval_epoch(self, data_loader, context_size=None):
        """Runs in eval mode on the given data loader and uses the whole time series as target."""
        epoch_mse = 0
        epoch_nll = 0
        if context_size is None:
            context_size = randint(*self.num_context_range)

        self.neural_process.eval()
        for i, data in enumerate( tqdm(data_loader)):
            with torch.no_grad():
                x, y = data
                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, use_y0=self.use_y0)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                # Use the whole time series as target.
                x_target = x.to(self.device)
                y_target = y.to(self.device)
                p_y_pred = self.neural_process(x_context, y_context, x_target, y_target, y0)

                nll = self._loss(p_y_pred, y_target)
                epoch_nll += nll.cpu().item()

                mse = ((y_target-p_y_pred.mean)**2).mean()
                epoch_mse += mse.item()

        epoch_mse = epoch_mse / len(data_loader)
        epoch_nll = epoch_nll / len(data_loader)
        self.epoch_mse_history.append(epoch_mse)
        self.epoch_logp_history.append(epoch_nll)

        return epoch_mse, epoch_nll, x_context, y_context, x_target, y_target, p_y_pred # only output the last batch

    def _loss(self, p_y_pred, y_target, q_target=None, q_context=None):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        if isinstance(p_y_pred, Bernoulli):
            # Pixels might be in (0, 1), but we still treat them as binary
            # so this is a bit of a hack. This is needed because pytorch checks the argument
            # to log_prob is in the support of the Bernoulli distribution (i.e. it is 0 or 1).
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean(dim=0).sum()
        else:
            nll = -p_y_pred.log_prob(y_target).mean(dim=0).sum()

        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        if q_target is None and q_context is None:
            return nll

        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return nll + kl

class NeuralProcessTrainer:
    """
    Class to handle training of Neural Processes.
    Code adapted from https://github.com/EmilienDupont/neural-processes

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.TimeNeuralProcess, neural_process.NeuralODEProcess instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    """
    def __init__(self,
                 device: torch.device,
                 neural_process: TimeNeuralProcess,
                 optimizer: torch.optim.Optimizer,
                 num_context_range: Tuple[int, int],
                 num_extra_target_range: Tuple[int, int],
                 max_context=None,
                 use_all_targets=False,
                 use_y0=True):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.max_context = max_context
        self.use_all_targets = use_all_targets
        self.use_y0 = use_y0

        self.epoch_loss_history = []
        self.epoch_nfe_history = []
        self.epoch_mse_history = []
        self.epoch_logp_history = []

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs: int):
        """
        Trains Neural (ODE) Process.

        Parameters
        ----------
        train_data_loader : Data loader to use for training
        val_data_loader: Data loader to use for validation
        epochs: Number of epochs to train for
        """
        self.neural_process.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            epoch_loss = self.train_epoch(train_data_loader)
            # some bit about the self.epoch_nfe_history.append if we want to track nfe in training
            self.epoch_loss_history.append(epoch_loss)
            self.eval_epoch(val_data_loader)

    def train_epoch(self, data_loader, posweight):
        epoch_loss = 0.
        self.neural_process.train()
        p_y_all = []
        x_ct_all = []
        y_ct_all = []
        x_tar_all = []
        y_tar_all = []
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer.zero_grad()

            # Extract data
            x, y = data
            #################### added by sally for emotion dataset################
            # x, y = x.unsqueeze(0), y.unsqueeze(0)
            #################### added by sally for emotion dataset################
            points = x.size(1)

            # Sample number of context and target points
            num_context = randint(*self.num_context_range)
            num_extra_target = randint(*self.num_extra_target_range)
            if self.use_all_targets:
                num_extra_target = points - num_context

            ####################### added by sally ################################
            # num_context = points
            # num_extra_target = randint(*self.num_extra_target_range)
            # if self.use_all_targets:
            #     num_extra_target = points - num_context
            ####################### added by sally ################################

            # Create context and target points and apply neural process
            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))#x_context are the points of interpolation; and x_target are the points including both interpolation and extrapolation
            y0 = y0.to(self.device)#y0 is the first point of the sequence if given
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)

            ########### forcing initial to 0#####
            # y0 = torch.from_numpy(np.asarray([0])).unsqueeze(-1).to(self.device)
            # y_context[0] = 0
            ########### forcing initial to 0#####

            p_y_pred, q_target, q_context = (
                self.neural_process(x_context, y_context, x_target, y_target, y0))
            loss = self._loss(p_y_pred, y_target, q_target, q_context, posweight=torch.from_numpy(np.asarray(posweight)).to(device))#posweight is added by sally

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.cpu().item()
            p_y_all.append(p_y_pred)
            x_ct_all.append(x_context)
            y_ct_all.append(y_context)
            x_tar_all.append(x_target)
            y_tar_all.append(y_target)

        return epoch_loss / len(data_loader), x_ct_all, y_ct_all, x_tar_all, y_tar_all, p_y_all #only output the last batch

    def eval_epoch(self, data_loader, context_size=None):
        """Runs in eval mode on the given data loader and uses the whole time series as target."""
        epoch_mse = 0
        epoch_nll = 0
        if context_size is None:
            context_size = randint(*self.num_context_range)

        self.neural_process.eval()
        p_y_all = []
        x_ct_all = []
        y_ct_all = []
        x_tar_all = []
        y_tar_all = []

        for i, data in enumerate( tqdm(data_loader)):
            with torch.no_grad():
                x, y = data
                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, locations=np.arange(x.shape[1]), use_y0=self.use_y0)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                ########### forcing initial to 0#####
                # y0 = torch.from_numpy(np.asarray([0])).unsqueeze(-1).to(self.device)
                # y_context = torch.from_numpy(np.asarray([0])).unsqueeze(-1).unsqueeze(-1).to(self.device)
                ########### forcing initial to 0#####
                # Use the whole time series as target.
                x_target = x.to(self.device)
                y_target = y.to(self.device)
                p_y_pred = self.neural_process(x_context, y_context, x_target, y_target, y0)

                # nll = self._loss(p_y_pred, y_target,posweight=torch.from_numpy(np.asarray(1)).to(device))# posweight added by Sally
                # epoch_nll += nll.cpu().item()
                #
                # mse = ((y_target-p_y_pred.mean)**2).mean()
                # epoch_mse += mse.item()

                p_y_all.append(p_y_pred)
                x_ct_all.append(x_context)
                y_ct_all.append(y_context)
                x_tar_all.append(x_target)
                y_tar_all.append(y_target)

        # epoch_mse = epoch_mse / len(data_loader)
        # epoch_nll = epoch_nll / len(data_loader)
        # self.epoch_mse_history.append(epoch_mse)
        # self.epoch_logp_history.append(epoch_nll)

        return epoch_mse, epoch_nll, x_ct_all, y_ct_all, x_tar_all, y_tar_all, p_y_all  # only output the last batch

    def _loss(self, p_y_pred, y_target, q_target=None, q_context=None, posweight = None):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        if isinstance(p_y_pred, Bernoulli):
            # Pixels might be in (0, 1), but we still treat them as binary
            # so this is a bit of a hack. This is needed because pytorch checks the argument
            # to log_prob is in the support of the Bernoulli distribution (i.e. it is 0 or 1).
            pred = p_y_pred.logits
            ##loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            # loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight = posweight)
            # nll = loss(pred, y_target).mean(dim=0).sum()
            if len(torch.unique(y_target)) > 1 and y_target.shape[1]>1:
                mlayer = torch.nn.Sigmoid()
                nll = -1 * scipy.stats.pointbiserialr(mlayer(pred)[0,:,0].detach().numpy(), y_target[0,:,0].detach().numpy())[0]
                nll = torch.from_numpy(np.asarray(nll)).to(device)
            else:
                loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight = posweight)
                nll = loss(pred, y_target).mean(dim=0).sum()
        else:
            nll = -p_y_pred.log_prob(y_target).mean(dim=0).sum()

        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        if q_target is None and q_context is None:
            return nll

        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return nll + kl
        #return 0.1*nll + 0.9*kl
        #return nll
