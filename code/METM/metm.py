import torch
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class METM(nn.Module):
    def __init__(self, args, vocab_size, num_sources, sources_embeddings, word_embeddings=None):
        super(METM, self).__init__()

        ## define hyperparameters
        self.num_topics = args.num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = args.t_hidden_size
        self.rho_size = args.rho_size
        self.enc_drop = args.enc_drop
        self.emsize = args.emb_size
        self.num_sources = num_sources
        self.t_drop = nn.Dropout(args.enc_drop)

        self.theta_act = self.get_activation(args.theta_act)
        
        ## define the word embedding matrix \rho
        if args.train_word_embeddings:
            self.rho = nn.Linear(self.rho_size, self.vocab_size, bias=False)
        else:
            num_embeddings, emsize = word_embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = word_embeddings.clone().float().to(device)

        # source lambda --> L X L for S sources
        if args.train_source_embeddings:
            self.source_lambda = nn.Parameter(sources_embeddings)
        else:
            self.source_lambda = sources_embeddings.clone().float().to(device)
        
        ## define the matrix containing the topic embeddings
        # alpha --> L X K 
        self.alpha = nn.Linear(self.rho_size, self.num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))
        
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(self.vocab_size, self.t_hidden_size), 
                self.theta_act,
                nn.Linear(self.t_hidden_size, self.t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)

    def get_activation(self, act):
        """Returns the activation function
        """
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

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

        # get beta at specific source s, topic k, time t
    def get_beta_sk(self, alpha, s, k):
        """Returns the full topic matrix beta of shape S x K x V
        """
        # alpha: L x K
        # source_lambda: S x L

        # 1 x L -> L
        alpha_k = alpha[k,:]
        # print("alpha_k size="+str(alpha_k.shape))
        # print("lambda size="+str(self.source_lambda.shape))
        # print("lambda_s size="+str(self.source_lambda[s:,].shape))

        alpha_sk = alpha_k * self.source_lambda[s]

        # 1 x L prod L x V = L x V
        #print("alpha_sk shape="+str(alpha_sk.shape))
        #print("rho shape ="+str(self.rho.shape))
        logit = torch.matmul(alpha_sk, self.rho.permute(1, 0))
        #print("logit shape="+str(logit.shape))

        return F.softmax(logit, dim=-1) # 1 x V

    def get_beta_unique(self, uniq_tokens, uniq_sources):
        """Returns the topic matrix beta of shape S x K x T x V
        """
        # alpha: K x L
        # source_lambda: S x L

        # K x L -> 1 x K x L -> S' x K x L
        alpha_s = self.alpha.weight.unsqueeze(0).repeat(uniq_sources.shape[0], 1, 1)# alpha_s has shape 1 x K x L

        # S x L -> S' x L
        source_lambda_s = self.source_lambda[uniq_sources.type('torch.LongTensor')]

        # S' x L -> S' x 1 x L -> S' x K x L
        source_lambda_s = source_lambda_s.unsqueeze(1).repeat(1, self.num_topics, 1) # source_lambda has shape S' x 1 x L and after repeat it becomes S' x K x L

        alpha_s = alpha_s * source_lambda_s # S' x K x L

        #tmp = alpha_s.view(alpha_s.size(0)*alpha_s.size(1)*alpha_s.size(2), self.rho_size) # (S' x K x T') x L

        # (S' x K) x L prod L x V = (S' x K) x V
        logit = torch.matmul(alpha_s, self.rho.permute(1, 0))

        #logit = logit.view(alpha_s.size(0), alpha_s.size(1), alpha_s.size(2), -1) # S' x K x T' x V
        beta = F.softmax(logit, dim=-1)[:,:,uniq_tokens.type('torch.LongTensor')] # S' x K x V'
        return beta



    def get_beta(self):
        """Returns topic matrix of size S x K x V
           alphs size -> L X K
           source_lambda size -> S X L
        """
        alpha_s = self.alpha.weight.unsqueeze(0).repeat(self.num_sources, 1, 1) # alpha_s has shape 1 x K x L
        alpha_s = alpha_s * self.source_lambda.unsqueeze(1).repeat(1,self.num_topics,1) # source_lambda has shape S x 1 x L and after repeat it becomes S x K x L.
        #alpha_s = alpha_s.reshape(self.num_sources, self.num_topics, self.emsize)
        # alpha_s now becomes S x K x L

        # S x K x L prod L x V = S x K x V
        #logit = torch.mm(alpha_s.view(alpha_s.size(0)*alpha_s.size(1)*alpha_s.size(2), 
        #    self.rho_size), self.rho.permute(1, 0))
        try:
            logit = torch.matmul(alpha_s, self.rho.weight.permute(1,0))
        except:
            logit = torch.matmul(alpha_s, self.rho.permute(1,0))

        #logit = logit.view(alpha_s.size(0), alpha_s.size(1), alpha_s.size(2), -1) # S x T x K x V

        beta = F.softmax(logit, dim=-1) # S x K x V        
        return beta

        # try:
        #     layer = self.alpha
        #     logit = layer(self.rho.weight) # torch.mm(self.rho, self.alphas)
        # except:
        #     layer = self.alpha
        #     logit = layer(self.rho)
        # beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        # return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.matmul(theta, beta)
        preds = torch.log(res+1e-6)
        return preds 

    def forward(self, uniq_tokens, bows, normalized_bows, sources, num_docs, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta   
        unique_sources = sources.unique()
        unique_sources_idx = torch.cat([(unique_sources == source).nonzero()[0] for source in sources])

        beta = self.get_beta_unique(uniq_tokens, unique_sources) # S' x K x V'

        beta = beta[unique_sources_idx, :, :] # D x K x V

        ## get prediction loss
        preds = self.decode(theta, beta)
        #recon_loss = -(preds * bows).sum(1)
        recon_loss = -(preds * bows[:,uniq_tokens]).sum(1)

        
        if aggregate:
            recon_loss = recon_loss.mean()
            #total_loss /= len(countries)

        return recon_loss, kld_theta

