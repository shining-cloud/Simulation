import numpy as np 
import matplotlib.pyplot as plt #3.3.4 needed, won't work with any other versions of mpl
from scipy.stats import norm

graphiques = True #graphiques = True pour afficher les graphs de la question 3
save_graph = True #True pour sauvegarder les graphs en png, False sinon


"""----------------- 1. Implement vol function -----------------"""

def vol(a,b,v,rho,K,f,T): 
    z = v/a * (f*K)**((1-b)/2) * np.log(f/K)
    X = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho)/(1-rho))
    numer = a * (1 + (((1-b)**2)/24 * (a**2)/((f*K)**(1-b)) + 
                      (1/4) * (rho*b*v*a)/((f*K)**((1-b)/2)) + ((2-3*(rho**2))/24)*(v**2)) * T)     
    denom = (1 + (1-b)**2/24 * np.log(f/K)**2 + (1-b)**4/1920*np.log(f/K)**4) * (f*K)**((1-b)/2)
    return (numer/denom * z/X)


"""----------------- 2. Plot the vol surface and smile -----------------"""
T = 2 #years 
TTM =  0.119 #1 month 
alpha = 0.3 #a
beta = 1 #b
nu = 0.4 #v
rho = 0.1 #rho
price = 15 

# x and y for surface 
x = np.linspace(0,2,50)
y = np.linspace(price*1.5,price*0.5,1000)
X, Y = np.meshgrid(x,y)

# x for smile
x1 = np.linspace(price*0.5,price*1.5,100)

#2.1 Vol surface 
Z = vol(alpha,beta,nu,rho,Y,price,X)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Time To Maturity')
ax.set_ylabel('Strike')
ax.set_zlabel('vol')
ax.view_init(25, 200)
ax.set_title(f"alpha = {alpha}, beta = {beta}, nu = {nu}, rho ={rho}, price = {price}")
if save_graph:
    plt.savefig('volsurface.png',dpi=300)
plt.show()

#2.2 1-month vol smile 

y1 = vol(alpha,beta,nu,rho,x1,price,TTM)

plt.figure()
fig, ax = plt.subplots()
ax.plot(x1,y1, 'r')
ax.set_title(f'alpha = {alpha}, beta = {beta}, nu = {nu}, \nrho ={rho}, price = {price},TTM = {TTM} years')
ax.set_ylabel('vol')
ax.set_xlabel('Strike')
plt.tight_layout()
if save_graph:
    plt.savefig(f'smile TTM = {TTM}.png',dpi=300)
plt.show()   
    
    
"""----------------- 3 Imapct of parameters on smile -----------------"""



if graphiques:
#3.1.1 alpha surface
    Z1alpha = vol(alpha,beta,nu,rho,Y,price,X)
    Z2alpha = vol(alpha*1.2,beta,nu,rho,Y,price,X)
    Z0alpha = vol(alpha*0.8,beta,nu,rho,Y,price,X)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Time To Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('vol')
    ax.view_init(25, 200)
    ax.set_title(f"alpha = {alpha}, beta = {beta}, nu = {nu}, rho ={rho}, price = {price}")
    s2 = ax.plot_surface(X, Y, Z2alpha, label = 'alpha * 1.2')
    s1 = ax.plot_surface(X, Y, Z1alpha, label = f'alpha = {alpha}')
    s0 = ax.plot_surface(X, Y, Z0alpha, label = 'alpha * 0.8')
    s2._facecolors2d=s2._facecolor3d
    s2._edgecolors2d=s2._edgecolor3d
    s1._facecolors2d=s1._facecolor3d
    s1._edgecolors2d=s1._edgecolor3d
    s0._facecolors2d=s0._facecolor3d
    s0._edgecolors2d=s0._edgecolor3d
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_graph:
        plt.savefig('alpha_impact.png',dpi=300)
    plt.show()
    
#3.1.2 alpha smile 
    y1alpha = vol(alpha,beta,nu,rho,x1,price,TTM)
    y2alpha = vol(alpha*1.2,beta,nu,rho,x1,price,TTM)
    y0alpha = vol(alpha*0.8,beta,nu,rho,x1,price,TTM)
    
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x1,y0alpha, 'r',color='g',label = 'alpha * 0.8')
    ax.plot(x1,y1alpha, 'r',color = 'b',label = f"alpha = {alpha}")
    ax.plot(x1,y2alpha, 'r',label = 'alpha * 1.2')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(f'alpha = {alpha}, beta={beta}, nu = {nu}, \nrho ={rho}, price = {price},TTM = {TTM} years')
    ax.set_ylabel('vol')
    ax.set_xlabel('Strike')
    plt.tight_layout()
    if save_graph:
        plt.savefig(f'alpha_impact TTM = {TTM}.png',dpi=300)
    plt.show()   
    
    
#3.2 beta 
#3.2.1 beta surface
    Zbeta05 = vol(alpha,1/2,nu,rho,Y,price,X)
    Zbeta1 = vol(alpha,1,nu,rho,Y,price,X)
    Zbeta0 = vol(alpha,0,nu,rho,Y,price,X)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Time To Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('vol')
    ax.view_init(25, 200)
    ax.set_title(f"alpha = {alpha}, beta, nu = {nu}, rho ={rho}, price = {price}")
    s2 = ax.plot_surface(X, Y, Zbeta1, label = 'beta = 1')
    s1 = ax.plot_surface(X, Y, Zbeta05, label = 'beta = 0.5')
    s0 = ax.plot_surface(X, Y, Zbeta0, label = 'beta = 0')
    s2._facecolors2d=s1._facecolor3d
    s2._edgecolors2d=s1._edgecolor3d
    s1._facecolors2d=s1._facecolor3d
    s1._edgecolors2d=s1._edgecolor3d
    s0._facecolors2d=s0._facecolor3d
    s0._edgecolors2d=s0._edgecolor3d
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_graph:
        plt.savefig('beta_impact.png',dpi=300)
    plt.show()
    
#3.2.2 beta smile 
    y1beta = vol(alpha,0.5,nu,rho,x1,price,TTM)
    y2beta = vol(alpha,1,nu,rho,x1,price,TTM)
    y0beta = vol(alpha,0,nu,rho,x1,price,TTM)
    y3beta = vol(alpha,0.4,nu,rho,x1,price,TTM)
    
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x1,y2beta, 'r',label = 'beta = 1')
    ax.plot(x1,y1beta, 'r',color = 'b',label = "beta = 0.5")
    ax.plot(x1,y3beta, 'r',color='orange',label = 'beta = 0.4')
    ax.plot(x1,y0beta, 'r',color='g',label = 'beta = 0')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(f'alpha = {alpha}, beta, nu = {nu}, \n rho ={rho}, price = {price},TTM = {TTM} years')
    ax.set_ylabel('vol')
    ax.set_xlabel('Strike')
    plt.tight_layout()
    if save_graph:
        plt.savefig(f'beta_impact TTM = {TTM}.png',dpi=300)
    plt.show()   
    
    
#3.3.1 rho surface 
    Z1rho = vol(alpha,beta,nu,rho,Y,price,X)
    Z2rho = vol(alpha,beta,nu,rho*2,Y,price,X)
    Z0rho = vol(alpha,beta,nu,-rho,Y,price,X)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Time To Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('vol')
    ax.view_init(25, 200)
    ax.set_title(f"alpha = {alpha}, beta = {beta}, nu = {nu}, rho ={rho}, price = {price}")
    s2 = ax.plot_surface(X, Y, Z2rho, label = 'rho * 2')
    s1 = ax.plot_surface(X, Y, Z1rho, label = f'rho = {rho}')
    s0 = ax.plot_surface(X, Y, Z0rho, label = 'rho * -1')
    s2._facecolors2d=s2._facecolor3d
    s2._edgecolors2d=s2._edgecolor3d
    s1._facecolors2d=s1._facecolor3d
    s1._edgecolors2d=s1._edgecolor3d
    s0._facecolors2d=s0._facecolor3d
    s0._edgecolors2d=s0._edgecolor3d
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_graph:
        plt.savefig('rho_impact.png',dpi=300)
    plt.show()  
    
#3.3.2 rho smile 
    y1rho = vol(alpha,beta,nu,rho,x1,price,TTM)
    y2rho = vol(alpha,beta,nu,rho*2,x1,price,TTM)
    y0rho = vol(alpha,beta,nu,rho*-1,x1,price,TTM)
    y3rho = vol(alpha,beta,nu,rho*3,x1,price,TTM)
    y00rho = vol(alpha,beta,nu,rho*-2,x1,price,TTM)
    
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x1,y3rho, 'r',color='red',label = 'rho * 3')
    ax.plot(x1,y2rho, 'r',color='orange',label = 'rho * 2')
    ax.plot(x1,y1rho, 'r',color = 'black',label = f"rho = {rho}")
    ax.plot(x1,y0rho, 'r',color='green',label = 'rho * -1')
    ax.plot(x1,y00rho, 'r',color='blue',label = 'rho * -2')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(f'alpha = {alpha}, beta={beta}, nu = {nu}, \nrho ={rho}, price = {price},TTM = {TTM} years')
    ax.set_ylabel('vol')
    ax.set_xlabel('Strike')
    plt.tight_layout()
    if save_graph:
        plt.savefig(f'rho_impact TTM = {TTM}.png',dpi=300)
    plt.show()    
        
#3.4.1 nu surface
    Z1nu = vol(alpha,beta,nu,rho,Y,price,X)
    Z2nu = vol(alpha,beta,nu*1.2,rho,Y,price,X)
    Z0nu = vol(alpha,beta,nu*0.8,rho,Y,price,X)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Time To Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('vol')
    ax.view_init(25, 200)
    ax.set_title(f"alpha = {alpha}, beta = {beta}, nu = {nu},\n rho ={rho}, price = {price}")
    s2 = ax.plot_surface(X, Y, Z2nu, label = 'nu * 1.2')
    s1 = ax.plot_surface(X, Y, Z1nu, label = f'nu = {nu}')
    s0 = ax.plot_surface(X, Y, Z0nu, label = 'nu * 0.8')
    s2._facecolors2d=s2._facecolor3d
    s2._edgecolors2d=s2._edgecolor3d
    s1._facecolors2d=s1._facecolor3d
    s1._edgecolors2d=s1._edgecolor3d
    s0._facecolors2d=s0._facecolor3d
    s0._edgecolors2d=s0._edgecolor3d
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_graph:
        plt.savefig('nu_impact.png',dpi=300)
    plt.show()   
    
#3.4.2 nu smile
    y1nu = vol(alpha,beta,nu,rho,x1,price,TTM)
    y2nu = vol(alpha,beta,nu*1.2,rho,x1,price,TTM)
    y0nu = vol(alpha,beta,nu*0.8,rho,x1,price,TTM)
    y3nu = vol(alpha,beta,nu*1.5,rho,x1,price,TTM)
    y00nu = vol(alpha,beta,nu*0.4,rho,x1,price,TTM)
    
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x1,y3nu, 'r',color='r',label = 'nu * 1.5')
    ax.plot(x1,y2nu, 'r',color='orange',label = 'nu * 1.2')
    ax.plot(x1,y1nu, 'r',color = 'black',label = f"nu = {nu}")
    ax.plot(x1,y0nu, 'r',color='g',label = 'nu * 0.8')
    ax.plot(x1,y00nu, 'r',color='blue',label = 'nu * 0.4')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(f'alpha = {alpha}, beta = {beta}, nu = {nu}, \nrho ={rho}, price = {price},TTM = {TTM} years')
    ax.set_ylabel('vol')
    ax.set_xlabel('Strike')
    plt.tight_layout()
    if save_graph:
        plt.savefig(f'nu_impact TTM = {TTM}.png',dpi=300)
    plt.show()     


""""4. MONTE CARLO SIMULATIONS"""
#4.1 Monte Carlo classique
def SABRMCprice2(T,f,K,a,b,v,rho,r,NPaths,NSteps):
    dt = T/NSteps
    mean = np.zeros(2)
    cov = np.array([[dt,rho*dt],[rho*dt,dt]])
    a_vec = a*np.ones([NSteps,NPaths])
    f_vec = f*np.ones([NSteps,NPaths]) 
    for t in range(NSteps-1):
        dW = np.random.multivariate_normal(mean,cov,NPaths)
        dW1 = dW[:,0]
        dW2 = dW[:,1]
        f_vec[t+1,:] = f_vec[t,:] + a_vec[t,:] * (f_vec[t,:] ** b) * dW2
        a_vec[t+1,:] = a_vec[t,:] +  v * a_vec[t,:] * dW1
        
    pv_vec = np.exp(-r*T)*np.maximum(f_vec[-1,:] - K ,0) 
    var_pv = np.var(pv_vec,ddof=1) #Unbiased variance estimator
    sigsqrtn = np.sqrt(var_pv)/np.sqrt(NPaths)
    pv = np.mean(pv_vec)
    
    i1 = pv - 1.96*sigsqrtn
    i2=  pv + 1.96*sigsqrtn
    print('----------------------')
    print ("lower boundary: ",i1)
    print ("Higher boundary: ",i2)
    return pv

#4.2 Monte Carlo with Antithetic variables
def SABRMCpriceAntithetic(T,f,K,a,b,v,rho,r,NPaths,NSteps):
    dt = T/NSteps
    mean = np.zeros(2)
    cov = np.array([[dt,rho*dt],[rho*dt,dt]])
    a_vec1 = a*np.ones([NSteps,NPaths])
    f_vec1 = f*np.ones([NSteps,NPaths])  
    a_vec2 = a_vec1
    f_vec2 = f_vec1
    for t in range(NSteps-1):
        dW = np.random.multivariate_normal(mean,cov,NPaths)
        dW1 = dW[:,0]
        dW2 = dW[:,1]
        
        f_vec1[t+1,:] = f_vec1[t,:] + a_vec1[t,:] * (f_vec1[t,:] ** b) * dW2
        a_vec1[t+1,:] = a_vec1[t,:] +  v * a_vec1[t,:] * dW1
        
        f_vec2[t+1,:] = f_vec2[t,:] + a_vec2[t,:] * (f_vec2[t,:] ** b) * (-dW2)
        a_vec2[t+1,:] = a_vec2[t,:] +  v * a_vec2[t,:] * (-dW1)
    payoff = 1/2 * (np.maximum(f_vec1[-1,:]-K ,0) + np.maximum(f_vec2[-1,:]-K ,0))
    pv_vec = np.exp(-r*T)*payoff
    varpv = np.var(pv_vec,ddof=1) #Unbiased variance estimator
    sigsqrtn =np.sqrt(varpv/NPaths) 
    pv = np.mean(pv_vec)
    
    i1 = pv - 1.96*sigsqrtn
    i2=  pv + 1.96*sigsqrtn
    
    print('----------------------')
    print ("lower boundary: ",i1)
    print ("Higher boundary: ",i2)
    return pv


#4.3 Analytical price 

def Analytical(T,f,K,a,b,v,rho,r):
    sigma = vol(a,b,v,rho,K,f,T)
    d1 = (np.log(f/K) + 1/2 * sigma**2 * T )/ (sigma * np.sqrt(T))
    d2 = (np.log(f/K) - 1/2 * sigma**2 * T )/ (sigma * np.sqrt(T))
    price = np.exp(-r*T)* ( f*norm.cdf(d1) - K * norm.cdf(d2))
    return price


""" ----------------- 5. Prices comparison -------------------- """

T = 2 
K = 15
alpha = 0.3 #a
beta = 1 #b
nu = 0.4 #v
rho = 0.1 #rho
r = 0.1 #r

NPaths = 100
NSteps = 300 * T #100 par an

ATM = K + 0.001 
ITM = K * 1.5
OTM = K * 0.5

# Print the prices 

print("\n --------- Analytical prices ---------")
print(f'ATM: {Analytical(T,ATM,K,alpha,beta,nu,rho,r)}')
print(f'ITM: {Analytical(T,ITM,K,alpha,beta,nu,rho,r)}')
print(f'OTM: {Analytical(T,OTM,K,alpha,beta,nu,rho,r)}')

#MC Price
print(f"\n ----- MC prices w/o Antithetic variables, {NPaths} simuls, {NSteps} steps-----")
print(f'ATM: {SABRMCprice2(T,ATM,K,alpha,beta,nu,rho,r,NPaths,NSteps)}')
print(f'ITM: {SABRMCprice2(T,ITM,K,alpha,beta,nu,rho,r,NPaths,NSteps)}')
print(f'OTM: {SABRMCprice2(T,OTM,K,alpha,beta,nu,rho,r,NPaths,NSteps)}')

#MC Antithetic Price
print(f"\n ----- MC prices w/ Antithetic variables, {NPaths} simuls, {NSteps} steps-----")
print(f'ATM: {SABRMCpriceAntithetic(T,ATM,K,alpha,beta,nu,rho,r,NPaths,NSteps)}')
print(f'ITM: {SABRMCpriceAntithetic(T,ITM,K,alpha,beta,nu,rho,r,NPaths,NSteps)}')
print(f'OTM: {SABRMCpriceAntithetic(T,OTM,K,alpha,beta,nu,rho,r,NPaths,NSteps)}')