#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:01:06 2020

@author: yueli
"""

model.train()
acc_loss = 0
acc_nll = 0
acc_kl_theta_loss = 0
acc_kl_eta_loss = 0
acc_kl_alpha_loss = 0
cnt = 0
indices = torch.randperm(args.num_docs_train)
indices = torch.split(indices, args.batch_size) 

idx = 0
ind = indices[0]

optimizer.zero_grad()
model.zero_grad()
data_batch, times_batch, sources_batch = data.get_batch(
    train_tokens, train_counts, ind, args.vocab_size, train_sources, args.emb_size, temporal=True, times=train_times)
sums = data_batch.sum(1).unsqueeze(1)
if args.bow_norm:
    normalized_data_batch = data_batch / sums
else:
    normalized_data_batch = data_batch

# forward
bows=normalized_data_batch
bows=data_batch
normalized_bows=normalized_data_batch
times=times_batch
rnn_inp=train_rnn_inp
num_docs=args.num_docs_train

sources = sources_batch


# def forward(self, bows, normalized_bows, times, rnn_inp, num_docs)
bsz = normalized_bows.size(0)
coeff = num_docs / bsz 
alpha, kl_alpha = model.get_alpha()
eta, kl_eta = model.get_eta(rnn_inp)
theta, kl_theta = model.get_theta(eta, normalized_bows, times)
kl_theta = kl_theta.sum() * coeff


# get_beta
betas = torch.zeros(model.num_sources, model.num_times, model.num_topics, model.vocab_size) # S x T x K x V

for i in range(model.num_sources):

    alpha_s = alpha * model.source_lambda[i] # T x K x L elem-prod 1 x L

    if model.train_word_embeddings:
        logit = model.rho(alpha_s.view(alpha_s.size(0) * alpha_s.size(1), model.rho_size))
    else:
        tmp = alpha_s.view(alpha_s.size(0)*alpha_s.size(1), model.rho_size) # (T x K) x L
        logit = torch.mm(tmp, model.rho.permute(1, 0)) # (T x K) x L prod L x V = (T x K) x V

    logit = logit.view(alpha.size(0), alpha.size(1),  -1) # T x K x V

    betas[i] = F.softmax(logit, dim=-1)


beta = betas[sources.type('torch.LongTensor'),times.type('torch.LongTensor'),:,:]

# in get_nll
theta = theta.unsqueeze(1)
loglik = torch.bmm(theta, beta).squeeze(1)
loglik = torch.log(loglik+1e-6)
nll = -loglik * bows
nll = nll.sum(-1)


nll = nll.sum() * coeff
nelbo = nll + kl_alpha + kl_eta + kl_theta




















    