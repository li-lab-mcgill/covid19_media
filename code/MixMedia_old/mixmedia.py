"""This file defines a dynamic etm object.
"""

import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn
from torch.autograd import Variable

# from IPython.core.debugger import set_trace
from pdb import set_trace

# q_theta models
from transformers import ElectraModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinCosPosEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, device='cpu'):
        super(SinCosPosEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.device = device
        
    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False).to(self.device)

class Attention(nn.Module):
    def __init__(self, d_model, query_in_size, dropout_rate=0):
        super().__init__()
        # self.query_map = nn.Linear(query_in_size, d_model)
        self.W = nn.Parameter(torch.randn(d_model, d_model))    # (M * M)
        self.bias = nn.Parameter(torch.randn(1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate != 0 else None

    def forward(self, key, query, value):
        # query = F.tanh(self.query_map(query))
        if query.shape[0] > 1:  # not using a single vector as query for all samples in batch
            query_W_key = torch.bmm(torch.matmul(query, self.W), key.transpose(-2, -1)) # (B * N * N)
        else:
            query_W_key = torch.matmul(key, torch.matmul(query, self.W).transpose(-2, -1)).transpose(-2, -1)
        if self.dropout:
            query_W_key = self.dropout(query_W_key)
        weights = F.softmax(query_W_key + self.bias, dim=-1)  # (B * N * N)
        return weights, torch.bmm(weights, value)

class MixMedia(nn.Module):
    def __init__(self, args, word_embeddings):
        super(MixMedia, self).__init__()

        ## define hyperparameters
        self.num_topics = args.num_topics
        self.num_times = args.num_times
        self.vocab_size = args.vocab_size
        self.t_hidden_size = args.t_hidden_size
        self.eta_hidden_size = args.eta_hidden_size
        self.rho_size = args.rho_size
        self.emsize = args.emb_size
        self.enc_drop = args.enc_drop
        self.eta_nlayers = args.eta_nlayers
        self.t_drop = nn.Dropout(args.enc_drop)
        self.delta = args.delta
        self.train_embeddings = args.train_embeddings

        self.predict_labels = args.predict_labels
        self.multiclass_labels = args.multiclass_labels

        self.predict_cnpi = args.predict_cnpi

        self.num_sources = args.num_sources
        self.num_labels = args.num_labels

        self.theta_act = self.get_activation(args.theta_act)

        # LSTM params for q_theta
        # self.one_hot_qtheta_emb = args.one_hot_qtheta_emb
        # self.q_theta_arc = args.q_theta_arc
        # self.q_theta_layers = args.q_theta_layers
        # self.q_theta_input_dim = args.q_theta_input_dim
        # self.q_theta_hidden_size = args.q_theta_hidden_size
        # self.q_theta_bi = bool(args.q_theta_bi)
        # self.q_theta_drop = args.q_theta_drop

        # params for cnpi prediction
        self.cnpi_hidden_size = args.cnpi_hidden_size
        self.cnpi_drop = args.cnpi_drop
        self.cnpi_layers = args.cnpi_layers
        self.num_cnpis = args.num_cnpis
        self.use_doc_labels = args.use_doc_labels

        ## define the word embedding matrix \rho: L x V
        if args.train_embeddings:
            self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False).to(device) # L x V
        else:
            num_embeddings, emsize = word_embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            rho.weight.data = word_embeddings
            self.rho = rho.weight.data.clone().float().to(device)
    

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x L
        self.mu_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.rho_size)).to(device)
        self.logsigma_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.rho_size)).to(device)
    
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(args.vocab_size+args.num_topics, args.t_hidden_size), 
                    self.theta_act,
                    nn.Linear(args.t_hidden_size, args.t_hidden_size),
                    self.theta_act,
                )
        # if self.one_hot_qtheta_emb and self.q_theta_arc != 'electra':
        #     self.q_theta_emb = nn.Embedding(self.q_theta_input_dim, args.rho_size).to('cpu')
        #     self.q_theta_input_dim = self.rho_size  # change q_theta input size to rho_size after embedding
        #     # if args.q_theta_arc == 'trm':
        #     #     self.q_theta = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.rho_size, nhead=args.q_theta_heads, dropout=self.q_theta_drop), \
        #     #         self.q_theta_layers).to(device)
        #     #     q_theta_out_dim = self.rho_size
        #     # else:
        #     #     self.q_theta = nn.LSTM(self.rho_size, hidden_size=self.q_theta_hidden_size, \
        #     #         bidirectional=self.q_theta_bi, dropout=self.q_theta_drop, num_layers=self.q_theta_layers, batch_first=True).to(device)
        # # else:
        # if self.q_theta_arc == 'trm':
        #     self.pos_encode = SinCosPosEncoding(d_model=self.q_theta_input_dim, device=device)  # positional encoding
        #     self.q_theta = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.q_theta_input_dim, nhead=args.q_theta_heads, dropout=self.q_theta_drop), \
        #         self.q_theta_layers).to(device)
        #     q_theta_out_dim = self.q_theta_input_dim
        # elif self.q_theta_arc == 'lstm':
        #     self.q_theta = nn.LSTM(self.q_theta_input_dim, hidden_size=self.q_theta_hidden_size, \
        #         bidirectional=self.q_theta_bi, dropout=self.q_theta_drop, num_layers=self.q_theta_layers, batch_first=True).to(device)
        #     q_theta_out_dim = 2 * self.q_theta_hidden_size if self.q_theta_bi else self.q_theta_hidden_size
        # else:
        #     self.q_theta = ElectraModel.from_pretrained('google/electra-small-discriminator').to(device)
        #     q_theta_out_dim = self.q_theta.config.hidden_size
        # self.q_theta_att_query = nn.Parameter(torch.randn(1, q_theta_out_dim)).to(device)
        # self.q_theta_att = Attention(q_theta_out_dim, args.num_topics, self.q_theta_drop).to(device)
        self.mu_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        # self.mu_q_theta = nn.Linear(q_theta_out_dim + args.num_topics, args.num_topics, bias=True).to(device)
        # self.mu_q_theta = nn.Linear(q_theta_out_dim + args.num_topics, args.num_topics, bias=True).to(device)
        self.logsigma_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        # self.logsigma_q_theta = nn.Linear(q_theta_out_dim + args.num_topics, args.num_topics, bias=True).to(device)
        # self.logsigma_q_theta = nn.Linear(q_theta_out_dim + args.num_topics, args.num_topics, bias=True).to(device)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(args.vocab_size, args.eta_hidden_size).to(device)
        
        self.q_eta = nn.LSTM(args.eta_hidden_size, args.eta_hidden_size, args.eta_nlayers, dropout=args.eta_dropout, batch_first=True).to(device)

        self.mu_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True).to(device)
        self.logsigma_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True).to(device)
        
        self.max_logsigma_t = 10
        self.min_logsigma_t = -10


        ## define supervised component for predicting labels
        self.classifier = nn.Linear(args.num_topics, args.num_labels, bias=True).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        # predicting country-level npi
        if self.predict_cnpi:
            cnpi_input_size = args.num_topics + args.num_cnpis if self.use_doc_labels else args.num_topics
            self.cnpi_lstm = nn.LSTM(cnpi_input_size, hidden_size=self.cnpi_hidden_size, \
                bidirectional=False, dropout=self.cnpi_drop, num_layers=self.cnpi_layers, batch_first=True).to(device)
            self.cnpi_out = nn.Linear(self.cnpi_hidden_size, args.num_cnpis, bias=True).to(device)
            self.cnpi_criterion = nn.BCEWithLogitsLoss(reduction='sum')

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_alpha(self): ## mean field

        alphas = torch.zeros(self.num_topics, self.rho_size).to(device)
        kl_alpha = []

        alphas = self.reparameterize(self.mu_q_alpha, self.logsigma_q_alpha)

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(device)

        kl_alpha = self.get_kl(self.mu_q_alpha, self.logsigma_q_alpha, p_mu_0, logsigma_p_0)

        return alphas, kl_alpha.sum() # K x L


    # ## get source-specific etas
    def get_eta(self, rnn_inp): ## structured amortized inference

        etas = torch.zeros(self.num_sources, self.num_times, self.num_topics).to(device)
        kl_eta = []

        inp = self.q_eta_map(rnn_inp.view(rnn_inp.size(0)*rnn_inp.size(1), -1)).view(rnn_inp.size(0),rnn_inp.size(1),-1)
        
        hidden = self.init_hidden()

        output, _ = self.q_eta(inp, hidden)

        inp_0 = torch.cat([output[:,0,:], torch.zeros(self.num_sources,self.num_topics).to(device)], dim=1)

        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        
        etas[:, 0, :] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_sources, self.num_topics).to(device)
        logsigma_p_0 = torch.zeros(self.num_sources, self.num_topics).to(device)

        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)

        for t in range(1, self.num_times):

            inp_t = torch.cat([output[:,t,:], etas[:, t-1, :]], dim=1)

            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)

            if (logsigma_t > self.max_logsigma_t).sum() > 0:
                logsigma_t[logsigma_t > self.max_logsigma_t] = self.max_logsigma_t
            elif (logsigma_t < self.min_logsigma_t).sum() > 0:
                logsigma_t[logsigma_t < self.min_logsigma_t] = self.min_logsigma_t

            etas[:, t, :] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[:, t-1, :]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_sources, self.num_topics).to(device))

            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)

        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta



    def get_theta(self, eta, bows, times, sources): ## amortized inference
        """Returns the topic proportions.
        """
        eta_std = eta[sources.type('torch.LongTensor'), times.type('torch.LongTensor')] # D x K
        inp = torch.cat([bows, eta_std], dim=1)
        q_theta = self.q_theta(inp)
        # if self.one_hot_qtheta_emb and self.q_theta_arc != 'electra':
        #     embs = self.q_theta_emb(embs)
        # if self.q_theta_arc == 'trm':
        #     embs = self.pos_encode(embs)
        #     q_theta_out = self.q_theta(embs)
        # else:
        #     q_theta_out = self.q_theta(embs)[0]
        
        # max-pooling and concat with eta_std to get q_theta
        # q_theta_out = self.q_theta_att(key=q_theta_out, query=self.q_theta_att_query, value=q_theta_out)[1].squeeze()
        # q_theta_out = torch.max(q_theta_out, dim=1)[0]
        # q_theta = torch.cat([q_theta_out, eta_std], dim=1)
        # q_theta = self.q_theta_att(key=q_theta_out, query=eta_std.unsqueeze(1), value=q_theta_out)[1].squeeze()
        # q_theta = torch.cat([torch.max(q_theta_out, dim=1)[0], eta_std], dim=1)

        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)

        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        if (logsigma_theta > self.max_logsigma_t).sum() > 0:
            logsigma_theta[logsigma_theta > self.max_logsigma_t] = self.max_logsigma_t
        elif (logsigma_theta < self.min_logsigma_t).sum() > 0:
            logsigma_theta[logsigma_theta < self.min_logsigma_t] = self.min_logsigma_t        

        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_std, torch.zeros(self.num_topics).to(device))
        return theta, kl_theta


    def get_beta(self, alpha):
        """Returns the topic matrix beta of shape K x V
        """
        if self.train_embeddings:            
            logit = self.rho(alpha)
        else:
            logit = torch.mm(alpha, self.rho.permute(1, 0)) 
        
        beta = F.softmax(logit, dim=-1)
        return beta 


    def get_nll(self, theta, beta, bows):        
        loglik = torch.log(torch.mm(theta, beta))
        nll = -loglik * bows        
        return nll.sum(-1)


    def get_prediction_loss(self, theta, labels):        

        # test code only
        # targets = torch.zeros(theta.size(0), self.num_labels)
        # for i in range(theta.size(0)):
        #     targets[i,labels[i].type('torch.LongTensor').item()] = 1
        # labels = targets

        outputs = self.classifier(theta)

        if self.multiclass_labels: # multi-class prediction loss as independent Bernoulli

            pred_loss = (-labels * F.log_softmax(outputs, dim=-1) - (1-labels) * torch.log(1-F.softmax(outputs, dim=-1))).sum()

        else: # single-label prediction
            
            pred_loss = self.criterion(outputs, labels.type('torch.LongTensor').to(device))

        return pred_loss    

    def get_cnpi_prediction_loss(self, eta, cnpi_data):
        # get unique combinations of (source, time) as indices
        # indices = torch.tensor(list(dict.fromkeys(list(zip(sources.tolist(), times.tolist())))), dtype=torch.long)
        # get corresponding etas
        # current_eta = eta[indices[:, 0], indices[:, 1]] # D' x K
        cnpis = cnpi_data['cnpis']
        cnpi_mask = cnpi_data['cnpi_mask']
        if self.use_doc_labels:
            # this function is only called in training
            predictions = self.cnpi_lstm(torch.cat([eta, cnpi_data['train_labels']], dim=-1))[0]
        else:
            predictions = self.cnpi_lstm(eta)[0]
        # predictions = torch.max(predictions, dim=1)[0]
        predictions = self.cnpi_out(predictions)
        return self.cnpi_criterion(predictions * cnpi_mask, cnpis * cnpi_mask)

    def forward(self, bows, normalized_bows, embs, times, sources, labels, cnpi_data, rnn_inp, num_docs):        

        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz
        alpha, kl_alpha = self.get_alpha()
        
        eta, kl_eta = self.get_eta(rnn_inp)

        # theta, kl_theta = self.get_theta(eta, embs, times, sources)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times, sources)
        kl_theta = kl_theta.sum() * coeff
        
        beta = self.get_beta(alpha)
        
        nll = self.get_nll(theta, beta, bows)

        nll = nll.sum() * coeff        

        pred_loss = torch.tensor(0.0)
        cnpi_pred_loss = torch.tensor(0.0)

        nelbo = nll + kl_alpha + kl_eta + kl_theta
        
        if self.predict_labels:
            pred_loss = self.get_prediction_loss(theta, labels) * coeff
            nelbo = nll + kl_alpha + kl_eta + kl_theta + pred_loss
        else:
            nelbo = nll + kl_alpha + kl_eta + kl_theta

        if self.predict_cnpi:
            cnpi_pred_loss = self.get_cnpi_prediction_loss(eta, cnpi_data)
            nelbo += cnpi_pred_loss
        
        return nelbo, nll, kl_alpha, kl_eta, kl_theta, pred_loss, cnpi_pred_loss


    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, self.num_sources, nhid), weight.new_zeros(nlayers, self.num_sources, nhid))




















