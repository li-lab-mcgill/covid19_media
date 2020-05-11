"""This file defines a dynamic etm object.
"""

import torch
import torch.nn.functional as F 
import numpy as np 
import math 
from torch.distributions import MultivariateNormal
from sqrtm import sqrtm

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DMETM(nn.Module):
    def __init__(self, args, embeddings, source_logcovar=None):
        super(DMETM, self).__init__()

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

        self.num_sources = args.num_sources

        self.theta_act = self.get_activation(args.theta_act)

        ## define the word embedding matrix \rho
        if args.train_embeddings:
            self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            rho.weight.data = embeddings
            self.rho = rho.weight.data.clone().float().to(device)


        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is S x K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(args.num_sources, args.num_topics, args.num_times, args.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(args.num_sources, args.num_topics, args.num_times, args.rho_size))


        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(args.vocab_size+args.num_topics, args.t_hidden_size), 
                    self.theta_act,
                    nn.Linear(args.t_hidden_size, args.t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)


        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(args.vocab_size, args.eta_hidden_size)
        self.q_eta = nn.LSTM(args.eta_hidden_size, args.eta_hidden_size, args.eta_nlayers, dropout=args.eta_dropout)
        self.mu_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)

        # set the S x S covariance over the S sources:
        self.source_logcovar = source_logcovar

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

    def reparameterize_mvn(self, mu, logcovar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = sqrtm(torch.exp(logcovar))
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


    # for the multivariate gaussian alpha
    def get_kl_mvn(self, q_mu_mvn, q_logsigma_mvn, p_mu_mvn, p_logsigma_mvn):
        """Returns KL( N(q_mu_mvn, q_logsigma_mvn) || N(p_mu_mvn, p_logsigma_mvn) ).
        """                
        sigma_q_sq = torch.exp(q_logsigma_mvn)
        sigma_p_sq = torch.exp(p_logsigma_mvn)
        sigma_q_sq_inv = torch.inverse(sigma_q_sq)

        p_mu_mvn_vec = p_mu_mvn.reshape(p_mu_mvn.shape[0],1)
        q_mu_mvn_vec = q_mu_mvn.reshape(q_mu_mvn.shape[0],1)

        return torch.logdet(sigma_q_sq) - torch.logdet(sigma_p_sq) - sigma_p_sq.shape[0] + \
            torch.trace(sigma_q_sq_inv.mm(sigma_p_sq)) + \
            (q_mu_mvn_vec - p_mu_mvn_vec).t().mm(sigma_q_sq_inv).mm(q_mu_mvn_vec - p_mu_mvn_vec)


    def get_alpha(self): ## mean field for multivariate Gaussian one dimension per source
        
        alphas = torch.zeros(self.num_sources, self.num_times, self.num_topics, self.rho_size).to(device)

        kl_alpha = []        
        
        # at time 0, for each topic and each embedding, sample from multivariate Gaussian with 
        # mean and covariance over all sources
        # alpha is S x K x T x L
        # dimensions: source, topics, time, embeddings
        for k in range(self.num_topics):
            for l in range(self.rho_size):
                alphas[:,k,0,l] = self.reparameterize_mvn(self.mu_q_alpha[:, k, 0, l].reshape(self.num_sources, 1), 
                    self.logsigma_q_alpha[:, k, 0, l].reshape(self.num_sources, self.num_sources))

        p_mu_0 = torch.zeros(self.num_sources, self.num_topics, self.rho_size).to(device)        
        logsigma_p_0 = self.source_logcovar # known covariance matrix

        kl_0 = 0        
        for k in range(self.num_topics):
            for l in range(self.rho_size):        
                kl_0 += self.get_kl_mvn(self.mu_q_alpha[:, k, 0, l], self.logsigma_q_alpha[:, k, 0, l], p_mu_0, logsigma_p_0)

        kl_alpha.append(kl_0)

        for t in range(1, self.num_times):
            kl_t = 0
            for k in range(self.num_topics):
                for l in range(self.rho_size):

                    alphas[:, k, t, l] = self.reparameterize(self.mu_q_alpha[:, k, t, l], self.logsigma_q_alpha[:, k, t, l])                     
                    p_mu_ktl = alphas[:,k,t-1,l]
                    logsigma_p_ktl = torch.log(self.delta * self.source_logcovar.to(device))

                    kl_t =+ self.get_kl_mvn(self.mu_q_alpha[:, k, t, l], self.logsigma_q_alpha[:, k, t, l], p_mu_ktl, logsigma_p_ktl)

            kl_alpha.append(kl_t)

        kl_alpha = torch.stack(kl_alpha).sum()

        return alphas, kl_alpha.sum()


    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def get_theta(self, eta, bows, times): ## amortized inference
        """Returns the topic proportions.
        """
        eta_td = eta[times.type('torch.LongTensor')]
        inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(device))
        return theta, kl_theta



    def get_beta(self, alpha):
        """Returns the topic matrix beta of shape S x (T x K) x V
        """
        betas = torch.zeros(self.num_sources, self.num_times, self.num_topics, self.rho_size).to(device)

        for s in range(self.num_sources):
            if self.train_embeddings:
                logit = self.rho(alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)) # (T x K) x V
            else:
                tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
                logit = torch.mm(tmp, self.rho.permute(1, 0)) # find out what permute does
            logit = logit.view(alpha.size(0), alpha.size(1), -1)
            betas[s] = F.softmax(logit, dim=-1)
        return betas 



    # negative log-likelihood (nll)
    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = loglik
        loglik = torch.log(loglik+1e-6)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll  

    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz 
        alpha, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        kl_theta = kl_theta.sum() * coeff

        betas = self.get_beta(alpha)
        beta = beta[times.type('torch.LongTensor')]
        nll = self.get_nll(theta, beta, bows)
        nll = nll.sum() * coeff
        nelbo = nll + kl_alpha + kl_eta + kl_theta
        return nelbo, nll, kl_alpha, kl_eta, kl_theta

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))






