#!/usr/bin/env python
# coding: utf-8

# The aim of this project is to record how to price a synthetic CDO with homogeneous credits in the reference portfolio.

'''
Method 1 - Tree

To price a synthetic CDO tranche or evaluate its value, the key part is to estimate the distribution of default loss on the reference portfolio. However, the composites of the underlying asset pool is very complicated so that it's very hard to obtain the distribution of default loss. The tree method assumes that the correlations of default across the obligors in the reference portfolio depends on a common factor  𝑋 , so we can first compute the distribution of default loss conditional on this common factor  𝑋 , and then integrate  𝑋  to get the unconditional distribution of default loss. For simplication, we first introduce some notations as following.

𝐴𝑖               Notional amount of the obligor i
𝑞𝑖(𝑡)             Risk-neutral default probability of obligor i before time t
𝑅𝑖               Recovery rate of obligor

Whether the company i will default on its credits depends on its assets value 𝑍𝑖. If the assets value drops below a threshold Zi¯ (Z-bar), it will default. In vasicek model, the company value can be described by a one factor gaussian copula. In this project, we use will use this model to compute the distribution of default loss as well. Company 𝑖's assets value can be modelled as follows:
                                                                
                                                                𝑍𝑖=(√1−𝛽𝑖)𝑋+(√1−𝛽𝑖)𝜖𝑖
                                                                
where 𝑋 is the common or market factor which represents the sysmetic risk and is common to all obligor. 𝜖𝑖 is the idiosyncratic risk factor specific to obligor i. Both 𝑋 and 𝜖𝑖 are a standard gaussian variable and the random variables 𝑋 and 𝜖1,...,𝜖𝑁  are assumed to be mutually independent. The factor loading 𝛽𝑖 is strictly constrained to lie in zero and one.

We define that default times follow a non-homogeneous Poisson process with the parameter 𝜆𝑡 which i s the hazard rate. If 𝜏 is a default time, we could have the survival probability of obligor 𝑖 between 0 and 𝑡 is :  
                                                                
                                                                𝑃(𝜏⩾𝑡)=exp−∫0-𝑡 𝜆(𝑢)𝑑𝑢
                                                                
If 𝜆𝑡 is flat beween 0 and 𝑡, it can be simplified as:

                                                                    𝑃(𝜏⩾𝑡)=exp(−𝜆𝑡)
                                                                    
If obligor 𝑖 defaults before time t, it means that its assets value drops below the threshold as well. We get:

                                                                 𝑞𝑖(𝑡)=𝑃(𝜏𝑖≤𝑡)=𝑃(𝑍𝑖<𝑍𝑖¯)                                                                   

Therefore, The conditional default probability 𝑞𝑖(𝑡|𝑋), which can be also expressed as 𝑃(𝑍𝑖<Zi¯|𝑋) is equal to

                                                           𝑞𝑖(𝑡|𝑋)=𝜙([𝜙^(−1)(𝑞𝑖(𝑡))−𝑋√𝛽𝑖]/(√1−𝛽𝑖))
                                                           
If the recovery rate and notional amount are constant for all obligors, the distribution of default loss will take exact 𝑁+1 discrete values and it will degenerate to the distribution of number of defaults. If they are not constant, the distribution of default loss will take more discrete values and it needs to be appropriately discretize to decrease the computational burden. In this project, we assume the obligors in the reference portfolio are homogeneous, which means that they share the same notional amount and recovery rate. With this assumption, the distribution of default loss can be easily obtained by the distribution of number of default multiply the loss given default.

From now on, the key is to get the distribution of number of default. It can be computed by a recursive way. We denote 𝑝𝐾(𝑙,𝑡|𝑋) as the exact 𝑙 defaults by time  𝑡 conditional on common factor 𝑋 for a reference portfolio with 𝐾 obligors only.  𝑙 can be one, two, and up to  𝐾 .

From the Gaussian copula we can find that, 𝑞𝑖(𝑡|𝑀), the risk-neutral default probability of obligor 𝑖 by time 𝑡 conditional on common factor 𝑋, is independent with default probability of other obligors. So if we consider on adding one obligor into the reference portfolio, the distribution of number of defaults for the new reference portfolio with 𝐾+1 obligors can be calculated as follows:

                                                            𝑝𝐾+1(0,𝑡|𝑋)=𝑝𝐾(0,𝑡|𝑋)×(1−𝑞𝐾+1(𝑡|𝑋))
 
                                               𝑝𝐾+1(𝑙,𝑡|𝑋)=𝑝𝐾(𝑙,𝑡|𝑋)×(1−𝑞𝐾+1(𝑡|𝑋))+𝑝𝐾(𝑙−1,𝑡|𝑋)×𝑞𝐾+1(𝑡|𝑋)
 
                                                             𝑝𝐾+1(𝐾+1,𝑡|𝑋)=𝑝𝐾(𝐾,𝑡|𝑋)×𝑞𝐾+1(𝑡|𝑋)
                                                             
We have already known that if 𝐾=0, 𝑝𝐾(0,𝑡|𝑋) = 1. We can start from this equation and plug the result into the recursion above to get the complete distribution of number of defaults for the reference portfolio with N obligors.

When we get the distribution of number of defaults conditional on common factor 𝑋 , we just need to integrate 𝑋 to get the unconditional distribution of number of defaults as follow:

                                                                𝑝(𝑙,𝑡)=∫∞−∞𝑝𝑁(𝑙,𝑡|𝑋)𝜑(𝑋)𝑑𝑋
                                                                
There is one thing to notice that the homegeneous obligors mean the recovery rate and notional amount are the same, but it doesn't require the hazard rate are also the same across different obligors. If the hazard rates are the same as well, it's very easy to get the distribution of number of defaults. Because the hazard rates are the same and the risk-neutral probability of number of defaults conditional on 𝑋 are independent across the obligors, we don't need to do the recursion and the conditonal distribution of number of defaults follow binomial distribution,  𝐵(𝑁,𝑞𝑖(𝑡|𝑋)) , where 𝑁 is the size of the reference portfolio and 𝑞𝑖(𝑡|𝑋) is the same for all obligor in this case.

The computation speed is faster by using binomial distribution to calculate the conditional and unconditional distribution of number of defaults than the recursion. But it can only be applied when the hazard rates are the same across the obligors in the reference portfolio. The codes below inclue both methods and the algorithm will use the recursion if the parameter "method" is set as True, otherwise it will use the binomial distribution. The results show that two methods are indifferent when the hazard rates are the same.

Once we obtain the unconditional distribution of number of defaults and default loss, we can price a specific synthetic CDO trance. The first step is to compute the expected loss of the tranche on each payment day. We denote 𝐸𝐿𝑗 as the expected loss of the tranche on payment day 𝑗 and it's equal to:

                                                             𝐸𝐿𝑖=∑𝑙=0-𝑁 𝑝(𝑙,𝑇𝑗)max(min(𝑙𝐴(1−𝑅),𝐻)−𝐿,0)
 or

                                                             𝐸𝐿𝑖=∑𝑙=0-𝑁 𝑝(𝑙,𝑇𝑗)min(max(𝑙𝐴(1−𝑅)−𝐿,0),𝐻−𝐿)  
                                                             
where 𝑝(𝑙,𝑇𝑗) is the unconditional distribution of number of defaults on payment day 𝑇𝑗. It should be noted that the unconditional distribution of number of defaults should be computed on each payment day. 𝐿 and 𝐻 are the lower and higher attachment point respectively.

The second step is to calculate the expected present value of the contingent leg and the fee leg. They can be computed as follow:

                                                                   𝐶𝑜𝑛𝑡𝑖𝑛𝑔𝑒𝑛𝑡=∑𝑗=1-𝑛 𝐷𝑗(𝐸𝐿𝑗−𝐸𝐿𝑗−1)                                                             
where 𝐷𝑗 is the risk-free discount factor for payment date 𝑗.

                                                                      𝐹𝑒𝑒=𝑠∑𝑗=1-𝑛 𝐷𝑗Δ𝑖(𝐻−𝐿−𝐸𝐿𝑗)
                                                                      
where Δ𝑗 is the accrual factor for payment data 𝑗 ( Δ𝑗≈𝑇𝑗−𝑇𝑗−1 ) and s is the annual spread paid to investors.

The mark-to-market value of the tranche is the difference between the expected present value of fee leg and contingent leg considering from the perspective of the CDO investors. It is computed as follow:

                                                                        𝑀𝑇𝑀=𝐹𝑒𝑒−𝐶𝑜𝑛𝑡𝑖𝑛𝑔𝑒𝑛𝑡   
 
When we price a synthetic CDO tranche at the inception, the mark-to-market value of the tranche should be zero which mean it is fair to both CDO issuer and the investors. Setting mark-to-market value to be zero, we can get the par spread of the CDO tranche:

                                                                   𝑠_𝑃𝑎𝑟=𝐶𝑜𝑛𝑡𝑖𝑛𝑔𝑒𝑛𝑡∑𝑛𝑗=1𝐷𝑗Δ𝑖(𝐻−𝐿−𝐸𝐿𝑗)

Note:In this pricing part, we constructed a synthetic CDO with 3 tranches, senior, mezzanine and equity tranche. The notional amount of the reference portfolio is 1 billion dollars with 100 homogeneous obligors. All the obligors have the following characteristics:

Single-name CDS spread                 60 basis points
Notional amount                        10 million dollars
Recovery rate                          40 percent
Default hazard rate                    1 percent per year
Asset correlation(𝛽i)                  30 percent

This CDO has a maturity of 5 years with quarterly payments. The risk-free rate is set to be 5 percent per year(continously compounded). All else assumptions keep the same as above.

We first calculated conditional and unconditional probabilities of number of defaults on 1 year as an example and then calculate the par spread of each tranche of this hypothetical CDO. The table below shows that the number of defaults tends to be less with the common factor 𝑋 increases. The unconditional probabilities of default can be considered as a weighted average of the conditional probabilities, weighted by common factor 𝑋.
'''

# import packages
import numpy as np
from functools import lru_cache
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
np.set_printoptions(suppress=True)

# Function to compute the conditional unconditional default probability of the reference portfolio
def def_dist(N, t, beta, hazard_rate, X, method):
    # cumulative default probability of one credit in each payment day
    qi_t = 1 - np.exp(-hazard_rate * t)
    #generate threshold
    z_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    #conditional default probability
    qi_t_X = st.norm.cdf((z_bar - np.sqrt(beta) * X)/np.sqrt(1- beta), loc = 0, scale =1)
    con_def = np.zeros(shape = (N + 1, len(X)))
   
    # Save the calculation results in cache temporarily to increase the speed of recursion
    @lru_cache(maxsize=1024, typed=False)
    # Recursive Function
    def recur(num_credit, num_default):
        if num_credit == 0 and num_default == 0:
            con_prob = 1
        elif num_default == 0:
            con_prob = recur(num_credit -1, 0) * (1 - qi_t_X)
        elif num_credit == num_default:
            con_prob =  recur(num_credit -1, num_default -1) * qi_t_X
        else:
            con_prob = recur(num_credit - 1, num_default) * (1 - qi_t_X) + recur(num_credit - 1, num_default-1) * qi_t_X
        return con_prob
    
    # Use recursion to get the default distribution conditional on common factor X
    if method == True:
        for i in range(0, N+1):
            con_def[i] = recur(N, i)
        uncon_def = np.dot(con_def, st.norm.pdf(X).reshape(len(X),1) *0.01).reshape(N + 1)
            
    # Use binomal distribution to get the default distribution conditional on common factor X -> this methods can only be applied when the hazard rates of 
    # each credit in the reference portfolio are the same
    else:
        for i in range(0,N+1):
            con_def[i] = st.binom.pmf(i, N, qi_t_X)
        uncon_def = np.dot(con_def, st.norm.pdf(X).reshape(len(X),1) *0.01).reshape(N + 1)

    return con_def,  uncon_def

# Generate the conditional and unconditional default probabiliy on 1 year
Table_default_dist = pd.DataFrame(def_dist(100, 1, 0.3, 0.01, np.arange(-2, 3, 1), True)[0].round(3).tolist())
Table_default_dist = Table_default_dist.rename_axis('Number of defaults')
Table_default_dist.columns = ['X = -2', 'X = -1', 'X = 0', 'X = 1', 'X = 2']
Table_default_dist.loc[:,'Unconditional default distribution p(l,t)'] = def_dist(100, 1, 0.3, 0.01, np.arange(-5, 5, 0.01), True)[1].round(3).tolist()
Table_default_dist[Table_default_dist < 0.001] = '*'

Table_default_dist.head(20)

'''
The codes below shows the par spread of each tranche of this CDO. It can be found that the par spread of senior tranche is lowest and that of equity tranche is the highes. This result is consistent with the risk of the tranche which the investor would afford. Meanwhile, the weighted average par spread of this CDO is about 74 basis points, which is larger than the single-name CDS spread of the obligors(60 bps) in the reference portfolio. If the CDO issuer sells the CDO to investors at par spreads computed by the model and sell credit protection on the reference portfolio, this transaction will be negative carry for the CDO issuer and is not a perfect hedge, which means the underlying risk of the CDO is actually larger than the single-name CDS.

The CDO has an additional risk related to the timing of default. Early and severely defaults will benefit the CDO issuer, because the equity tranche is wiped out very quickly and the issuer doesn't need to pay very high coupon to the investor. If the defaults all occur very late, it will have a negative impact on the issuer.
'''

# Function to calculate the par spread of a specified synthetic CDO tranche
def CDO_pricing_tree(notional, recovery_rate, L, H, N, T, hazard_rate, beta, pay_freq, rf, method):
    # Generate each payment date
    t = np.arange(0, T + 1/pay_freq, 1/pay_freq)
    # Simulate common factor X
    X = np.arange(-5, 5, 0.01)
    El = np.zeros(T * pay_freq + 1)
    Dt = np.exp(-rf * t)
    num_def = np.arange(0, N + 1 ,1)
    
    # Compute the expected loss of the tranche on each payment date
    for i in range(1, T * pay_freq + 1):
        p_l_t = def_dist(N, t[i], beta, hazard_rate, X, method)[1].reshape(N + 1)
        loss_tranche = np.maximum(np.minimum(num_def * notional * (1 - recovery_rate), H) - L, 0)
        El[i] =(p_l_t * loss_tranche).sum()
    
    # Compute fee leg and risk present value of 1 bp to obtain the par spread
    contingent = np.sum(Dt[1:] * np.diff(El))
    rpv_01 = np.sum(Dt[1:] * np.diff(t) *((H-L) - El[1:]))
    
    s_par = contingent/rpv_01
        
    return round(s_par * 10000), rpv_01, contingent

# Generate the par spread table
senior_price = CDO_pricing_tree(10, 0.4, 100, 1000, 100, 5, 0.01, 0.3, 4, 0.05, True)[0]
mezzanine_price = CDO_pricing_tree(10, 0.4, 30, 100, 100, 5, 0.01, 0.3, 4, 0.05, True)[0]
equity_price = CDO_pricing_tree(10, 0.4, 0, 30, 100, 5, 0.01, 0.3, 4, 0.05, True)[0]

Dict_CDO_price = {'Tranche':['Equity', 'Mezzanine', 'Senior', 'Entire porfolio'],
                     'Attachment points (percent)':['0-3', '3-10', '10-100', '0-100'],
                     'Notional amount ($ millions)':[30, 70, 900, '{:,}'.format(1000)],
                     'Par spread':[equity_price, mezzanine_price, senior_price, 60]}

Table_CDO_price = pd.DataFrame(Dict_CDO_price)
Table_CDO_price

'''
Method 2 - Monte Carlo Simulation 1

The synthetic CDO tranche can also be priced by Monte Carlo simulation without compute the default probabilities of reference portfolio. We still assumes that the assets value of obligor 𝑖 is modelled by one factor Gaussian copula. From the discusion above we know that the probability the obligor 𝑖 will default before time 𝑡  is equal to the probabilities of the assets value of obligor  𝑖  drop below the threshold  Zi¯ . We can get solve this threshold by:

                                                                           Zi¯=𝜙^(−1)[𝑞𝑖(𝑡)]
                                                                           
Therefore the simulation algorithm can be summarized by the following steps:

Step 1: Initialize  𝐿𝑛𝑡𝑗=0 , where  𝑡𝑗  is the payment date and 𝑛 is the n𝑡ℎ simulation
Step 2: Simulate common factor 𝑋 from standard Gaussian distribution
Step 3: For each obligor i, compute its assets value                                                                          
                                                                         𝑍𝑖=(√𝛽𝑖)𝑋+(√1−𝛽𝑖)𝜖𝑖 
Step 4: If  𝑍𝑖<Zi¯,  
                                                                            𝐿𝑛𝑡𝑗=𝐿𝑛𝑡𝑗+𝐿𝐺𝐷𝑖 
Step 5: Compute the loss of the tranche Step 6: Repeat step 1 to step 5 to generate enough losses of the tranche
Step 7: Compute the expected loss of the tranche on payment date 𝑡𝑗 by:

                                                             𝐸(𝐿𝑡𝑗^(L,H))≈ 1/𝑁mc∑𝑛=1-𝑁mc min(max(𝐿𝑛𝑡𝑗−𝐿,0),𝐻−𝐿)
 
Step 8: Repeat step 1 to step 7 to get the expected losses of tranche on each payment date

When we get the expected lossed of tranche on each payment date, we can calculate the calculate the par spread by the formula in the part above. The following codes show this simulation algorithm and the results show that this simulation method will give very similar par spreads as the tree method.
'''

# Generate simulation algorithm
def CDO_pricing_MC_X(notional, recovery_rate, L, H, N, T, hazard_rate, beta, pay_freq, rf):
    t = np.arange(0, T + 1/pay_freq, 1/pay_freq)
    qi_t = 1 - np.exp(-hazard_rate * t)
    z_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    # Generation simulation 
    n_MC = 100000
    n_period = T * pay_freq + 1
    El_t = np.zeros(n_period)
    Dt = np.exp(-rf * t)
    a_i = np.ones(N) * np.sqrt(beta)
    
    for i in range(n_period):
        bi = np.sqrt(1 - beta) * np.random.normal(0, 1, N * n_MC)
        bi = bi.reshape(N,n_MC)
        X = np.random.normal(0,1,n_MC).reshape(1, n_MC)
        z_i = np.dot(a_i.reshape(N, 1), X) + bi
        num_def = (z_i <= z_bar[i]).sum(axis = 0)
        
        El = num_def * notional * (1 - recovery_rate)
        El = np.minimum(np.maximum(El - L, 0), H-L)
        El_t[i] = El.mean()
        
    rpv_01 = np.sum(Dt[1:] * np.diff(t) * ((H - L) - El_t[1:]))
    contingent = np.sum(Dt[1:] * np.diff(El_t))
    s_par = contingent / rpv_01
    return round(s_par * 10000)

print('Par spread of Senior tranche by MC(simulate X) is {} basis points'.format(CDO_pricing_MC_X(10, 0.4, 100, 1000, 100, 5, 0.01, 0.3, 4, 0.05)))
print('Par spread of Mezzanine tranche by MC(simulate X) is {} basis points'.format(CDO_pricing_MC_X(10, 0.4, 30, 100, 100, 5, 0.01, 0.3, 4, 0.05)))
print('Par spread of Equity tranche by MC(simulate X) is {} basis points'.format(CDO_pricing_MC_X(10, 0.4, 0, 30, 100, 5, 0.01, 0.3, 4, 0.05)))

'''
Method 2 - Monte Carlo Simulation 2

The simulation method above is to obtain the par spreads by simulating the common factor  𝑋 . We know that the assets value of obligor  𝑖  is modelled by one factor Gaussian copula. With the charateristics of the Gaussian copula, we can find the assets value of the obligors  𝑍=[𝑍𝑖,...,𝑍𝑁]  follows multivariate normal distribution with mean,  𝑢=[0,...,0]^𝑇 , and covariance matrix,  𝑐𝑜𝑣=[1, √𝛽𝑖𝛽𝑗
                                                                     ....
                                                                   √𝛽𝑖𝛽𝑗, 1]. 

So we can skip the step 1 to 3 and simulate the assets values of obligors directly and then proceed to the following steps. The code below shows the logic of this simulation method. The results are consistents with the 2 methods above(tree and simulation by 𝑋).
'''
# Generate simulation algorithm
def CDO_pricing_MC_mv(notional, recovery_rate, L, H, N, T, hazard_rate, beta, pay_freq, rf):
    t = np.arange(0, T + 1/pay_freq, 1/pay_freq)
    qi_t = 1 - np.exp(-hazard_rate * t)
    Dt = np.exp(-rf * t)
    z_bar = st.norm.ppf(qi_t, loc = 0, scale = 1)
    n_period = T * pay_freq + 1
    n_MC = 100000
    El_t = np.zeros(n_period)
    mean = np.zeros(N)
    # Generate covariance matrix
    cov = np.ones(N*N).reshape(N,N) * beta
    for i, j in zip(range(0,N), range(0,N)):
        cov[i,j] = 1
    # Simulate Z and compute the expected loss on each payment day
    for i in range(n_period):
        Z = np.random.multivariate_normal(mean, cov, n_MC)
        Z = (Z <= z_bar[i])
        num_def = Z.sum(axis = 1)
        El = num_def * notional * (1 - recovery_rate)
        El = El.reshape(1, n_MC)
        El = np.minimum(np.maximum(El - L, 0), H-L)
        El_t[i] = El.mean()
        
    rpv_01 = np.sum(Dt[1:] * np.diff(t) * ((H - L) - El_t[1:]))
    contingent = np.sum(Dt[1:] * np.diff(El_t))
    s_par = contingent / rpv_01
    return round(s_par * 10000)

print('Par spread of Senior tranche by MC(simulate Z) is {} basis points'.format(CDO_pricing_MC_mv(10, 0.4, 100, 1000, 100, 5, 0.01, 0.3, 4, 0.05)))
print('Par spread of Mezzanine tranche by MC(simulate Z) is {} basis points'.format(CDO_pricing_MC_mv(10, 0.4, 30, 100, 100, 5, 0.01, 0.3, 4, 0.05)))
print('Par spread of Equity tranche by MC(simulate Z) is {} basis points'.format(CDO_pricing_MC_mv(10, 0.4, 0, 30, 100, 5, 0.01, 0.3, 4, 0.05)))

'''
Correlation Risk of the CDO Tranche

Correlations across the obligors in the reference portfolio are the important parameter when we price a synthetic CDO tranche. At inception, the mark-to-market value of CDO is zero, this is based on correlation at inception. When the correlation changes during the CDO's life, its mark-to-market value will change as well. So CDO has the correlation risk. In the hypothetical CDO we constructed above, we obtained the par spread by setting the correlation equals 0.3. In this part we will let the assets correlation varies between 0 and 0.9 to check how the CDO price changes.

The figure below shows that the mark-to-market value or the price of equity tranche increases with the increase of correlation, but the price of the senior tranche decreases. Normally, the default of reference portfolio will not impact on the senior tranche a lot because the equity and mezzanine tranche absorb the losses first. But the probabilities of large defaults will increases as the correlation increases. The equity and mezzanine tranche will be wiped out quickly and senior tranche has to take the losses. Therefore, the value of senior tranche will decrease. However, under the high correlation, not only will the probability of very large defaults increase, but also the probability of very few defaults will increases as well. The investor of equity tranche will benefit more from very few defaults than they loss from very large defaults, because they have already taken most of the risk of CDO and their situation will not be much more worse. So the total effect will be positive for the equity tranche investor and the price of equity tranche will increase as correlation increases.
'''

# Define revaluation algorithm
def CDO_pricing_value(notional, recovery_rate, L, H, N, T, hazard_rate, beta, base_beta, pay_freq, rf, method):
    
    rpv_01 = CDO_pricing_tree(notional, recovery_rate, L, H, N, T, hazard_rate, beta, pay_freq, rf, method)[1]
    contingent = CDO_pricing_tree(notional, recovery_rate, L, H, N, T, hazard_rate, beta, pay_freq, rf, method)[2]
    
    s_par = CDO_pricing_tree(notional, recovery_rate, L, H, N, T, hazard_rate, base_beta, pay_freq, rf, method)[0]
    
    value = s_par/10000 * rpv_01 - contingent
    
    return (value)

# Generate the price of CDO tranche after correlation changes
corr = np.arange(0, 1,0.1)
senior_p =[]
equity_p =[]

for i in range(10):
    senior = CDO_pricing_value(10, 0.4, 100, 1000, 100, 5, 0.01, corr[i], 0.3, 4, 0.05, True)
    equity = CDO_pricing_value(10, 0.4, 0, 30, 100, 5, 0.01, corr[i], 0.3, 4, 0.05, True)
    senior_p.append(senior)
    equity_p.append(equity)

# Generate the plot of effect of correlation on CDO tranche
plt.figure(figsize = (6,4))    
plt.plot(corr, senior_p, color = 'blue', label = 'Senior', marker = 'o')
plt.plot(corr, equity_p, color = 'red', label = 'Equity', marker = 'x')
plt.title('Effect of Correlation on CDO Tranches')
plt.xlim(0,0.9)
plt.ylim(-15,15)
plt.xticks(np.arange(0,1,0.1))
plt.yticks(range(-15,20,5))
plt.xlabel('Correlation')
plt.ylabel('Value of CDO ($ million)')
plt.legend()
plt.show()





