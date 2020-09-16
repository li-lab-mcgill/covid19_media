#/usr/bin/python

import os
import torch
import torch.nn.functional as F
from torch import nn, optim
# from torch.utils.data import DataLoader

import pytorch_lightning as pl

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

    # model configs
    parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--hidden_size', type=int, default=128, help='rnn hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_layers', type=int, default=1, help='number of rnn layers')

    return parser.parse_args()

class RNN_CNPI_BaseModel(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.mapping = nn.Linear(self.configs['vocab_size'], self.configs['emb_size'])

        self.rnn = nn.LSTM(self.configs['emb_size'], hidden_size=self.configs['hidden_size'], bidirectional=False, \
            dropout=self.configs['dropout'], num_layers=self.configs['num_layers'], batch_first=True)

        self.rnn_out = nn.Linear(self.configs['hidden_size'], self.configs['num_cnpis'], bias=True)

    def forward(self, bows):
        # bows: batch_size x times_span x vocab_size
        bows_mapped = self.mapping(bows.view(-1, bows.shape[-1])).view(bows.shape[0], bows.shape[1], -1)
        # embs: batch_size x times_span x embedding_size
        rnn_hidden = self.rnn(embs)[0]
        return self.rnn_out(rnn_hidden)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.configs['lr'], weight_decay=self.configs['wdecay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10, min_lr=1e-7)
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
            return (torch.gather(labels, 1, idxs).sum(1) / labels.sum(1)).mean().item()
        else:
            return (torch.gather(labels, 1, idxs).sum(1) / k).mean().item()
    
    def validation_step(self, batch, batch_idx):
        batch_bows, batch_labels, batch_mask = batch
        batch_predictions = self(batch_bows)
        batch_mask = 1 - batch_mask
        batch_predictions_masked = batch_predictions * batch_mask
        batch_labels_masked = batch_labels * batch_mask

        results = pl.EvalResult()

        val_loss = F.binary_cross_entropy_with_logits(batch_predictions_masked, batch_labels_masked)
        results.log('val_loss', val_loss)

        top_k_recalls = {
            1: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 1),
            3: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 3),
            5: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 5),
            10: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 10),
            }
        results.log_dict({f"recall/{key}": value for key, value in top_k_recalls.items()})
        top_k_precs = {
            1: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 1, 'prec'),
            3: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 3, 'prec'),
            5: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 5, 'prec'),
            10: sefl.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 10, 'prec'),
            }
        results.log_dict({f"prec/{key}": value for key, value in top_k_precs.items()})
        top_k_f1s = {
            k: [(2 * top_k_recalls[k][0] * top_k_precs[k][0]) / \
                (top_k_recalls[k][0] + top_k_precs[k][0])] for k in [1, 3, 5, 10]
            }
        results.log_dict({f"f1/{key}": value for key, value in top_k_f1s.items()})
        return results

if __name__ == '__main__':
    args = parse_args()
    configs = vars(args)

    ## set seed
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)

    # initiate data module
    data_module = COVID_Data_Module(configs)

    # initiate model
    model = RNN_CNPI_BaseModel(configs)

    # train
    trainer = pl.Trainer(gradient_clip_val=args.clip, max_epochs=args.epochs, gpus=1)
    trainer.fit(model, dm)