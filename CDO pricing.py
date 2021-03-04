#!/usr/bin/env python
# coding: utf-8

# In[432]:


import numpy as np
from functools import lru_cache
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
np.set_printoptions(suppress=True)


# In[640]:


# Monte Carlo Simulation

def CDO_pricing_MC_homo(N, notional, T, pay_freq, L, H, beta, hazard_rate, recovery_rate, rf):
    ti = np.arange(0, T + 1/pay_freq, 1/pay_freq)

    qi_t = 1 - np.exp(-hazard_rate * ti)
   
    z_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    
    n_MC = 10000000
    n_period = T * pay_freq + 1
    El_ti = np.zeros(n_period)
    Di = np.exp(-rf * ti)
    a_i = np.ones(N) * np.sqrt(beta)
    bi = np.sqrt(1 - beta) * np.random.normal(0, 1, N)
    bi = bi.reshape(N,1)
    
    def loss_filter(x, lower, higher):
        tran_loss = np.min([np.max([x - lower, 0]), higher - lower])
        return tran_loss
    
    
    for i in range(n_period):
        X = np.random.normal(0, 1, n_MC).reshape(1, n_MC)
        z_i = np.dot(a_i.reshape(N, 1), X) + bi
        z_i = (z_i <= z_bar[i]).sum(axis = 0)
        z_i = z_i.reshape(1, n_MC)
        E_i = z_i * notional * (1 - recovery_rate)
        E_i = np.apply_along_axis(loss_filter, 0, E_i, lower = L, higher = H)
        El_ti[i] = np.mean(E_i)
        
    fee = np.sum(Di[1:] * np.diff(ti) * ((H - L) - El_ti[1:]))
    contingent = np.sum(Di[1:] * np.diff(El_ti))
    s_par = contingent / fee
    return round(s_par * 10000)
        


# In[641]:


CDO_pricing_MC_homo(100, 10, 5, 4, 100, 1000, 0.3, 0.01, 0.4, 0.05)


# In[621]:


# Monte Carlo Simulation

def CDO_pricing_MC_homo_basic(N, notional, T, pay_freq, L, H, beta, hazard_rate, recovery_rate, rf):
    ti = np.arange(0, T + 1/pay_freq, 1/pay_freq)

    qi_t = 1 - np.exp(-hazard_rate * ti)
   
    z_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    
    n_MC = 10000
    n_period = T * pay_freq + 1
    El_ti = np.zeros(n_period)
    Di = np.exp(-rf * ti)
    ep = np.random.normal(0,1,100)
    
    for i in range(n_period):
        loss = 0
        for j in range(0, 10000):
            loss_sim = 0
            X = np.random.normal(0,1)
            for k in range(0,100):
                z = np.sqrt(beta) * X + np.sqrt(1-beta) * ep[k]
                if z <= z_bar[i]:
                    loss_sim += notional * (1 - recovery_rate)
            loss_sim = np.min([np.max([loss_sim - L, 0]), H - L])
            loss += loss_sim
        El_ti[i] = loss/n_MC
        
    fee = np.sum(Di[1:] * np.diff(ti) * ((H - L) - El_ti[1:]))
    contingent = np.sum(Di[1:] * np.diff(El_ti))
    s_par = contingent / fee
    return round(s_par * 10000)
        


# In[639]:


CDO_pricing_MC_homo(100, 10, 5, 4, 0, 30, 0.3, 0.01, 0.4, 0.05)


# In[632]:


def CDO_improve(N, notional, T, pay_freq, L, H, ai, hazard_rate, recovery_rate, rf):
    ti = np.arange(0, T + 1/pay_freq, 1/pay_freq)
    qi_t = 1 - np.exp(-hazard_rate * ti)
    xi_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    Di = np.exp(-rf*ti)
    El_ti = np.zeros(21)
#     M = np.random.normal(0,1,10000)
    for i in range(0,21):
        loss_m = 0
        M = np.random.normal(0,1,500000)
        for j in range(0,500000):
            qi_t_M = st.norm.cdf((xi_bar[i] - ai * M[j])/np.sqrt(1 - ai**2), loc = 0, scale = 1)
            loss = np.random.binomial(100, qi_t_M) * notional * (1 - recovery_rate)
            loss = np.min([np.max([loss - L, 0]), H-L])
            loss_m += loss
        El_ti[i] = loss_m/500000
        
    fee = np.sum(Di[1:] * np.diff(ti) * ((H - L) - El_ti[1:]))
    contingent = np.sum(Di[1:] * np.diff(El_ti))
    s_par = contingent / fee
    return round(s_par * 10000)
        
    


# In[633]:


CDO(100, 10, 5, 4, 0, 30, np.sqrt(0.3), 0.01, 0.4, 0.05)


# Part.2

# In[397]:


def con_def_dist(N, t, ai, hazard_rate, M, method):
    qi_t = 1 - np.exp(-hazard_rate * t)
    xi_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    k = st.norm.cdf((xi_bar - ai*M)/np.sqrt(1-ai**2), loc = 0, scale =1)
    con_def = np.ones(101)
    @lru_cache(maxsize=1024, typed=False)
    def recur(num_credit, num_default):
        if num_default == 0:
            con_prob = 1 * (1-k) ** num_credit
            return con_prob
        elif num_credit == num_default:
            con_prob = 1 * k ** num_credit
            return con_prob
        else:
            con_prob = recur(num_credit-1,num_default) * (1-k) + recur(num_credit-1, num_default-1) * k
            return con_prob
    # Recursive way to get the conditional default probability given M - Time consumping 
    if method == True:
        for i in range(0, N+1):
            con_def[i] = round(recur(N, i),3)
        return con_def
    # Combine the implied assumption in recursion and utilize the probablitiy mass function of binomial distribution
    # Faster than recursive way in the paper
    else:
        for i in range(0,N+1):
            con_def[i] = st.binom.pmf(i,N,k)
        return con_def
    
        


# In[398]:


Table_default_dist = pd.DataFrame()
for i in [-2,-1,0,1,2]:
    con_def = pd.Series(con_def_dist(100, 1, np.sqrt(0.3), 0.01, i, False)).round(3)
    Table_default_dist = pd.concat([Table_default_dist, con_def], axis = 1)


# In[399]:


Table_default_dist = Table_default_dist.rename_axis('Number of defaults')
Table_default_dist.columns = ['M = -2', 'M = -1', 'M = 0', 'M = 1', 'M = 2']
uncon_default_dist = np.zeros(101)
for i in np.arange(-5,5, 0.01):
    uncon_default_dist += pd.Series(con_def_dist(100, 1, np.sqrt(0.3), 0.01, i, False)) * st.norm.pdf(i,0,1)*0.01
Table_default_dist.loc[:,'Unconditional default distribution p(l,t)'] = uncon_default_dist.round(3)


# In[400]:


Table_default_dist.head(20)


# In[401]:


def CDO_pricing(A, R, L, H, N, T, hazard_rate, ai, pay_freq, rf):
    ti = np.arange(0, T + 1/pay_freq, 1/pay_freq)
    qi_t = 1 - np.exp(-hazard_rate * ti)
    xi_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    M = np.arange(-5, 5, 0.01)
    EL_i = np.zeros(21)
    Di = np.exp(-rf * ti)
    for i in range(1, T * pay_freq + 1):
        for l in range(0, N + 1):
            qi_t_M = st.norm.cdf((xi_bar[i] - ai * M)/np.sqrt(1 - ai**2), loc = 0, scale = 1)
            p_l_t = np.sum(st.binom.pmf(l, 100, qi_t_M) * st.norm.pdf(M,loc = 0, scale = 1)*0.01)
            EL_i[i] += p_l_t * np.max([np.min([l * A *(1 - R), H]) - L, 0])
    contingent = np.sum(Di[1:] * np.diff(EL_i))
    fee = np.sum(Di[1:] * np.diff(ti) *((H-L) - EL_i[1:]))
    s_par = contingent/fee
    return round(s_par*10000)


# In[402]:


senior_price = CDO_pricing(10, 0.4, 100, 1000, 100, 5, 0.01, np.sqrt(0.3), 4, 0.05)
mezzanine_price = CDO_pricing(10, 0.4, 30, 100, 100, 5, 0.01, np.sqrt(0.3), 4, 0.05)
equity_price = CDO_pricing(10, 0.4, 0, 30, 100, 5, 0.01, np.sqrt(0.3), 4, 0.05)

Dict_CDO_price = {'Tranche':['Equity', 'Mezzanine', 'Senior', 'Entire porfolio'],
                     'Attachment points (percent)':['0-3', '3-10', '10-100', '0-100'],
                     'Notional amount ($ millions)':[30, 70, 900, '{:,}'.format(1000)],
                     'Par spread':[equity_price, mezzanine_price, senior_price, 60]}

Table_CDO_price = pd.DataFrame(Dict_CDO_price)
Table_CDO_price

