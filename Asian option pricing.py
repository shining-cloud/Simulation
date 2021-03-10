'''
This project includes:

1. Analytical solution of Asian option
2. Monte Carlo simulation to price an Asian option
3. Control variate technique
4. Comparision between classic Monte Carlo simulation and Monte Carlo simulation with control variate
'''

# import packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm


"""1. Classic MC"""
def AsianMC(T,S,K,sigma,r,NPaths,NSteps):
    if NPaths <= 1:
        print("2 sample paths are required at minimum")
        return
    dt = T/NSteps
    sqrtdt = np.sqrt(dt)
    S_vec = S*np.ones([NSteps+1,NPaths])
    for t in range(NSteps):
        dW = np.random.normal(0,sqrtdt,NPaths)
        S_vec[t+1,:] = S_vec[t,:] + r*S_vec[t,:]*dt  +  sigma * S_vec[t,:] * dW
    S_mean = np.mean(S_vec[1:,:],axis=0) #Don't keep S0, only 1 to NSteps
    payoff = np.maximum(S_mean - K ,0)
    
    pv_vec =  np.exp(-r*T) * payoff
    pv = np.mean(pv_vec)
    varpv = np.var(pv_vec,ddof=1)
    i1 = pv - 1.96*np.sqrt(varpv)/np.sqrt(NPaths)
    i2 = pv + 1.96*np.sqrt(varpv)/np.sqrt(NPaths)
    return i1,i2,pv,pv_vec



"""2. MC with CV"""
"""2.1  Implementing the analytical geometric average Asian option price """

#No dividends, m = t = t0 = 0
def AnalyticalGeom(T,S,K,sigma,r,NSteps):
    dt = T/NSteps
    
    v = r - 0.5*sigma**2 
    a = np.log(S) + v*dt + 0.5*v*(T-dt)
    b = sigma**2 * dt + (sigma**2 * (T-dt) )/(6*NSteps) * (2*NSteps -1)
    x = (a - np.log(K) + b) / np.sqrt(b)
    
    price = np.exp(-r*T) * (np.exp(a + 0.5*b) * norm.cdf(x) - K * norm.cdf(x-np.sqrt(b)))
    return price

"""2.2 Pilot simulation to estimate c*"""
def MCPilot(T,S,K,sigma,r,NPaths,NSteps): 
    if NPaths <= 1:
        print("2 sample paths are required at minimum")
        return
    dt = T/NSteps
    sqrtdt = np.sqrt(dt)
    S_vec = S*np.ones([NSteps+1,NPaths])
    for t in range(NSteps):
        dW = np.random.normal(0,sqrtdt,NPaths)
        S_vec[t+1,:] = S_vec[t,:] + r*S_vec[t,:]*dt  +  sigma * S_vec[t,:] * dW
    ari_price = np.exp(-r*T)*np.maximum(np.mean(S_vec[1:,:],axis=0)-K,0)
    geom_price = np.exp(-r*T)*np.maximum(np.exp(np.mean(np.log(S_vec[1:,:]),axis=0))-K , 0)
    varcovar = np.cov(ari_price,geom_price)
    return(-varcovar[0,1]/np.var(geom_price,ddof=1))

""" 2.3 Monte Carlo with CV """
def AsianMCCV(T,S,K,sigma,r,NPaths,NSteps):
    if NPaths <= 1:
        print("2 sample paths are required at minimum")
        return
    c_star = MCPilot(T,S,K,sigma,r,NPaths,NSteps)
    geom_anal = AnalyticalGeom(T,S,K,sigma,r,NSteps)
    
    dt = T/NSteps
    sqrtdt = np.sqrt(dt)
    S_vec = S*np.ones([NSteps+1,NPaths])
    for t in range(NSteps):
        dW = np.random.normal(0,sqrtdt,NPaths)
        S_vec[t+1,:] = S_vec[t,:] + r*S_vec[t,:]*dt  +  sigma * S_vec[t,:] * dW
    ari_price = np.exp(-r*T)*np.maximum(np.mean(S_vec[1:,:],axis=0)-K,0)
    geom_price = np.exp(-r*T)*np.maximum(np.exp(np.mean(np.log(S_vec[1:,:]),axis=0))-K , 0)
    
    pv_vec = ari_price + c_star * (geom_price - geom_anal)
    varprice = np.var(pv_vec,ddof=1)
    sigsqrtn = varprice/np.sqrt(NPaths)    
    pv = np.mean(pv_vec)
    i1 = pv - 1.96*sigsqrtn
    i2=  pv + 1.96*sigsqrtn
    return i1,i2,pv,pv_vec,c_star

""" 3. Price and confidence intervals comparisons """
T = 2
S = 20
K = 15
sigma = 0.3
r = 0.05 
NPaths = 10000
NSteps = 200 * T

MC = AsianMC(T,S,K,sigma,r,NPaths,NSteps)
MC_CV = AsianMCCV(T,S,K,sigma,r,NPaths,NSteps)
print('------------ Price w/o CV ------------')
print ("lower boundary: ",MC[0])
print ("Higher boundary: ",MC[1])
print("Price:",MC[2])


print('------------ Price w/ CV ------------')
print ("lower boundary: ",MC_CV[0])
print ("Higher boundary: ",MC_CV[1])
print("Price:",MC_CV[2])
print("c*: ",MC_CV[4])



fig,[ax1,ax2] = plt.subplots(nrows=1, ncols=2,figsize=(10,4))
ax1.hist(MC[3],bins = 50,color='red')
ax1.set_xlim(0, np.max(MC[3]))
ax1.set_title('Under classic MC method')
ax1.set_xlabel('Price')
ax1.set_ylabel('Frequency')

ax2.hist(MC_CV[3],bins = 50,color='blue')
ax2.set_xlim(0, np.max(MC[3]))
ax2.set_title('Under MC with CV method')
ax2.set_xlabel('Price')
ax2.set_ylabel('Frequency')

fig.suptitle(f'Simulated prices distribution with {NPaths} simulations')
fig.tight_layout()
plt.show()
