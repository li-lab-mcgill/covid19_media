"""This file defines a dynamic etm object.
"""

import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

# from IPython.core.debugger import set_trace
from pdb import set_trace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DMETM(nn.Module):
    def __init__(self, args, word_embeddings, sources_embeddings):
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
        self.train_word_embeddings = args.train_word_embeddings

        self.num_sources = args.num_sources

        self.theta_act = self.get_activation(args.theta_act)

        ## define the word embedding matrix \rho: L x V
        if args.train_word_embeddings:
            # self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False)
            self.rho = nn.Parameter(word_embeddings)
        else:
            num_embeddings, emsize = word_embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            rho.weight.data = word_embeddings
            self.rho = rho.weight.data.clone().float().to(device)
        
        ## define the source-specific embedding \lambda S x L' (DMETM)
        if args.train_source_embeddings:
            # self.source_lambda = nn.Parameter(torch.randn(args.num_sources, args.rho_size))
            # self.source_lambda = nn.Parameter(torch.ones(args.num_sources, args.rho_size))            
            self.source_lambda = nn.Parameter(sources_embeddings)
        else:
            # source_lambda = nn.Embedding(args.num_sources, args.rho_size)
            # source_lambda.weight.data = sources_embeddings
            # self.source_lambda = source_lambda.weight.data.clone().float().to(device)
            self.source_lambda = sources_embeddings.clone().float().to(device)

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
    
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
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            
            p_mu_t = alphas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()

        alphas = alphas.permute(1,0,2) # TxKxL -> KxTxL

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


    # incorporate source-specific embedding lambda
    def get_beta(self, alpha, uniq_tokens, uniq_sources, uniq_times):
        """Returns the topic matrix beta of shape S x K x T x V
        """
        # alpha: K x T x L
        # source_lambda: S x L
        
        # set_trace()

        # 1 x K x T' x L -> S x K x T' x L
        alpha_s = alpha[:,uniq_times.type('torch.LongTensor'),:]
        alpha_s = alpha_s.unsqueeze(0).repeat(uniq_sources.shape[0], 1, 1, 1)

        # S' x 1 x L -> S' x 1 x 1 x L -> S' x K x T x L
        source_lambda_s = self.source_lambda[uniq_sources.type('torch.LongTensor')]

        num_uniq_times = uniq_times.shape[0]
        
        # S' x 1 x 1 x L -> S' x K x T' x L
        source_lambda_s = source_lambda_s.unsqueeze(1).unsqueeze(1).repeat(1, self.num_topics, num_uniq_times,1)

        alpha_s = alpha_s * source_lambda_s # S' x K x T' x L

        tmp = alpha_s.view(alpha_s.size(0)*alpha_s.size(1)*alpha_s.size(2), self.rho_size) # (S' x T' x K) x L
        
        # (S' x T' x K) x L prod L x V' = (S' x T' x K) x V'
        logit = torch.mm(tmp, self.rho[uniq_tokens.type('torch.LongTensor'),:].permute(1, 0))

        logit = logit.view(alpha_s.size(0), alpha_s.size(1), alpha_s.size(2), -1) # S' x T 'x K x V'

        return F.softmax(logit, dim=-1) # S x K x T x V

    # get beta for full vocab (can be memory consuming for large vocab, time, source)
    def get_beta_full(self, alpha):
        """Returns the topic matrix beta of shape S x K x T x V
        """
        # alpha: K x T x L
        # source_lambda: S x L            

        # 1 x K x T x L -> S x K x T x L
        alpha_s = alpha.unsqueeze(0).repeat(self.num_sources, 1, 1, 1)

        # S x 1 x L -> S x 1 x 1 x L -> S x K x T x L
        source_lambda_s = self.source_lambda.unsqueeze(1).unsqueeze(1).repeat(1, self.num_topics, self.num_times, 1)

        alpha_s = alpha_s * source_lambda_s # S x K x T x L

        tmp = alpha_s.view(alpha_s.size(0)*alpha_s.size(1)*alpha_s.size(2), self.rho_size) # (S x T x K) x L
        
        # (S x T x K) x L prod L x V = (S x T x K) x V
        logit = torch.mm(tmp, self.rho.permute(1, 0))

        logit = logit.view(alpha_s.size(0), alpha_s.size(1), alpha_s.size(2), -1) # S x T x K x V

        return F.softmax(logit, dim=-1) # S x K x T x V


    # get beta at specific source s, topic k, time t
    def get_beta_skt(self, alpha, s, k, t):
        """Returns the full topic matrix beta of shape S x K x T x V
        """
        # alpha: K x T x L
        # source_lambda: S x L

        # 1 x 1 x L -> L
        alpha_kt = alpha[k,t,:].squeeze()

        alpha_skt = alpha_kt * self.source_lambda[s,:]

        # 1 x L prod L x V = L x V
        logit = torch.mm(alpha_skt.unsqueeze(0), self.rho.permute(1, 0))

        return F.softmax(logit, dim=-1) # 1 x V

    def check_beta(self, alpha, unique_tokens, times, sources):
        
        set_trace()
        
        beta_full_fast = self.get_beta_full(alpha) # S x K x T x V
                
        beta_full_slow = torch.ones(self.num_sources, self.num_topics, self.num_times, self.rho.size(0))

        for i in range(self.num_sources):
            for k in range(self.num_topics):
                for t in range(self.num_times):
                    beta_full_slow[int(i),int(k),int(t),:] = self.get_beta_skt(alpha, int(i),int(k),int(t))

        print((beta_full_fast - beta_full_slow).max())

        unique_sources = sources.unique()
        unique_times = times.unique()
        
        beta_fast_sel = self.get_beta(alpha, unique_tokens, unique_sources, unique_times) # S' x K x T' x V'

        beta_slow_sel = beta_full_slow[unique_sources.type('torch.LongTensor')]
        beta_slow_sel = beta_slow_sel[:,:,unique_times.type('torch.LongTensor'),:]
        beta_slow_sel = beta_slow_sel[:,:,:,unique_tokens.type('torch.LongTensor')]
        
        print((beta_fast_sel-beta_slow_sel).max())



        



    # def get_beta(self, alpha):
    #     """Returns the topic matrix \beta of shape T x K x V
    #     """
    #     if self.train_word_embeddings:
    #         logit = self.rho(alpha.view(alpha.size(0)*alpha.size(1), self.rho_size))
    #     else:
    #         tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
    #         logit = torch.mm(tmp, self.rho.permute(1, 0)) 
    #     logit = logit.view(alpha.size(0), alpha.size(1), -1)
    #     beta = F.softmax(logit, dim=-1)
    #     return beta 


    def get_nll(self, theta, beta, bows, unique_tokens):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)        
        loglik = torch.log(loglik+1e-6)
        nll = -loglik * bows[:,unique_tokens]
        nll = nll.sum(-1)
        return nll  

    def forward(self, unique_tokens, bows, normalized_bows, times, sources, rnn_inp, num_docs):

        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz         

        alpha, kl_alpha = self.get_alpha()

        eta, kl_eta = self.get_eta(rnn_inp)
        
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)

        kl_theta = kl_theta.sum() * coeff

        unique_sources = sources.unique()
        unique_sources_idx = torch.cat([(unique_sources == source).nonzero()[0] for source in sources])

        unique_times = times.unique()
        unique_times_idx = torch.cat([(unique_times == time).nonzero()[0] for time in times])
        
        beta = self.get_beta(alpha, unique_tokens, unique_sources, unique_times) # S' x K x T' x V'
        
        beta = beta[unique_sources_idx, :, unique_times_idx, :] # D' x K x V'

        nll = self.get_nll(theta, beta, bows, unique_tokens)
        
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




















