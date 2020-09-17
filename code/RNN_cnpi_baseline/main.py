#/usr/bin/python

import os, argparse, pickle, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
# from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data import COVID_Data_Module

def parse_args():
    parser = argparse.ArgumentParser(description='RNN CNPI prediction baseline')

    # data io params
    parser.add_argument('--dataset', type=str, help='name of corpus')
    parser.add_argument('--data_path', type=str, help='directory containing data')
    parser.add_argument('--save_path', type=str, help='path to save results')

    # training configs
    parser.add_argument('--batch_size', type=int, default=128, help='number of documents in a batch for training')
    parser.add_argument('--min_df', type=int, default=10, help='to get the right data..minimum document frequency')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--mode', type=str, default='train', help='train or eval model')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

    # model configs
    parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')
    parser.add_argument('--map_size', type=int, default=300, help='dimension of mapping')
    parser.add_argument('--hidden_size', type=int, default=128, help='rnn hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_layers', type=int, default=1, help='number of rnn layers')

    return parser.parse_args()

class RNN_CNPI_BaseModel(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.mapping = nn.Linear(self.configs['vocab_size'], self.configs['map_size'])

        self.rnn = nn.LSTM(self.configs['map_size'], hidden_size=self.configs['hidden_size'], bidirectional=False, \
            dropout=self.configs['dropout'], num_layers=self.configs['num_layers'], batch_first=True)

        self.rnn_out = nn.Linear(self.configs['hidden_size'], self.configs['num_cnpis'], bias=True)

    def forward(self, bows):
        # bows: batch_size x times_span x vocab_size
        bows_mapped = self.mapping(bows.view(-1, bows.shape[-1])).view(bows.shape[0], bows.shape[1], -1)
        # bows_mapped: batch_size x times_span x mapped_size
        rnn_hidden = self.rnn(bows_mapped)[0]
        return self.rnn_out(rnn_hidden)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.configs['lr'], weight_decay=self.configs['wdecay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10, min_lr=1e-7)
        scheduler = {
            'scheduler': scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_checkpoint_on'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch_bows, batch_labels, batch_mask = batch
        batch_predictions = self(batch_bows)
        return F.binary_cross_entropy_with_logits(batch_predictions * batch_mask, batch_labels * batch_mask)

    def compute_top_k_recall_prec(self, labels, predictions, k=5, metric='recall'):
        '''
        inputs:
        - labels: tensor, (number of samples, number of classes)
        - predictions: tensor, (number of samples, number of classes)
        - k
        - metric: recall or prec
        output:
        - top-k recall or precision of the batch
        '''
        assert metric in ['recall', 'prec'], 'metric is either recall or prec'

        # remove ones without positive labels
        has_pos_labels = labels.sum(1) != 0
        labels = labels[has_pos_labels, :]
        predictions = predictions[has_pos_labels, :]
        idxs = torch.argsort(predictions, dim=1, descending=True)[:, 0: k]
        if metric == 'recall':
            return (torch.gather(labels, 1, idxs).sum(1) / labels.sum(1)).mean()
        else:
            return (torch.gather(labels, 1, idxs).sum(1) / k).mean()
    
    def validation_step(self, batch, batch_idx):
        batch_bows, batch_labels, batch_mask = batch
        batch_predictions = self(batch_bows)
        batch_mask = 1 - batch_mask
        batch_predictions_masked = batch_predictions * batch_mask
        batch_labels_masked = batch_labels * batch_mask

        val_loss = F.binary_cross_entropy_with_logits(batch_predictions_masked, batch_labels_masked)

        results = pl.EvalResult(checkpoint_on=val_loss)
        results.log('val_loss', val_loss)

        top_k_recalls = {
            1: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 1),
            3: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 3),
            5: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 5),
            10: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 10),
            }
        results.log_dict({f"recall/{key}": value for key, value in top_k_recalls.items()})
        top_k_precs = {
            1: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 1, 'prec'),
            3: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 3, 'prec'),
            5: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 5, 'prec'),
            10: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 10, 'prec'),
            }
        results.log_dict({f"prec/{key}": value for key, value in top_k_precs.items()})
        top_k_f1s = {
            k: (2 * top_k_recalls[k] * top_k_precs[k]) / \
                (top_k_recalls[k] + top_k_precs[k]) for k in [1, 3, 5, 10]
            }
        results.log_dict({f"f1/{key}": value for key, value in top_k_f1s.items()})
        return results

if __name__ == '__main__':
    args = parse_args()
    configs = vars(args)

    time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    print(f"Experiment time stamp: {time_stamp}")

    if not os.path.exists(os.path.join(configs['save_path'], time_stamp)):
        os.makedirs(os.path.join(configs['save_path'], time_stamp))

    # save configs
    with open(os.path.join(configs['save_path'], time_stamp, 'configs.json'), 'w') as file:
        json.dump(configs, file)

    with open(os.path.join(configs['data_path'], 'min_df_{}'.format(configs['min_df']), 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    configs['vocab_size'] = len(vocab)
    del vocab

    with open(os.path.join(os.path.join(configs['data_path'], 'min_df_{}'.format(configs['min_df']), 'cnpis.pkl')), 'rb') as file:
        cnpis = pickle.load(file)
    configs['num_cnpis'] = cnpis.shape[-1]
    del cnpis

    ## set seed
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)

    # initiate data module
    data_module = COVID_Data_Module(configs)

    # initiate model
    model = RNN_CNPI_BaseModel(configs)

    # train
    tb_logger = pl_loggers.TensorBoardLogger(f"lightning_logs/{time_stamp}")
    trainer = pl.Trainer(
        gradient_clip_val=args.clip, 
        max_epochs=args.epochs, 
        gpus=1, 
        logger=tb_logger,
        weights_save_path=os.path.join(configs['save_path'], time_stamp)
        )
    trainer.fit(model, data_module)