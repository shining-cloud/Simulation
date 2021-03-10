# This project intends to change Y = X + theta from P measure to Q measure.
# X ~ N(0,1) under P measure to Y ~ N(0,1) under Q measure

# import packages
%matplotlib inline
import  matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as st

theta = 2
# Simulate X~N(0,1)
X = np.random.normal(0, 1, 100000)

# Generate Y~N(theta, 1)
Y = X+theta

# Generate Radon-Nikodym derivative by Y(w)
Z = np.exp((theta**2 - 2*theta * Y)/2)

#Generate Radon-Nikodym derivative by X(w)
# Z = np.exp((-(theta**2) - 2*theta * X)/2)

# Change measure: P-measure to Q-measure
dPy = st.norm.pdf(Y,theta, 1)
dQy = dPy * Z

# Generate plot of distributions under 2 measures
plt.scatter(Y,dQy, color = 'red')
sns.kdeplot(Y, color = 'blue')
plt.legend(['P-measure','Q-measure'])
plt.vlines(x = 2, ymin = 0, ymax = 0.39,linestyle = '--')
plt.vlines(x = 0, ymin = 0, ymax = 0.39, linestyle = '--')
plt.plot()

theta = 2
# Simulate X~N(0,1)
X = np.random.normal(0, 1, 100000)

# Generate Y~N(theta, 1)
Y = X+theta
Y = np.array(sorted(Y))

# Generate Radon-Nikodym derivative by Y(w)
Z = np.exp((theta**2 - 2*theta * Y)/2)

#Generate Radon-Nikodym derivative by X(w)
# Z = np.exp((-(theta**2) - 2*theta * X)/2)

# Change measure: P-measure to Q-measure
dPy = st.norm.pdf(Y,theta, 1)
dQy = dPy * Z

# Generate plot of distributions under 2 measures
plt.plot(Y,dQy, color = 'red')
sns.kdeplot(Y, color = 'blue')
plt.legend(['P-measure','Q-measure'])
plt.vlines(x = 2, ymin = 0, ymax = 0.39,linestyle = '--')
plt.vlines(x = 0, ymin = 0, ymax = 0.39, linestyle = '--')
plt.plot()
