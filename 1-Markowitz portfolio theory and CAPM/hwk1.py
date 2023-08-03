# -*- coding: utf-8 -*-
"""Hwk1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h2y6f-VbSIj_6K7FO2_R0g5VFV4QvNQr

# Math 584 - Homework 1
*Diane PERES*

---

### Importation
"""

import pandas as pd
import numpy as np 
import csv
import matplotlib.pyplot as plt
import math
#!pip install yfinance
import yfinance as yf
import scipy.optimize
from scipy import stats

#from google.colab import files
 
#uploaded = files.upload()

"""## 1. Construct the efficient frontier

### (a) Produce the estimated vector of **mean returns** (sample mean returns) and the estimated **covariance matrix** (sample covariance matrix) and print them.
"""

#read list of tickers from a csv file
df_bid = pd.read_csv('./TechTickers.csv', header=None)
tickers_file = 'TechTickers.csv'
tickers = []
f = open(tickers_file,"r",encoding='utf-8-sig')
for line in csv.reader(f):
    tickers.append(str(line[0]))
f.close
#print(tickers)
tickers_str = tickers[0]
for s in tickers[1:]: tickers_str=tickers_str+", "+s
print(tickers_str)

#downoad the prices and volumes for the previously read list of tickers for the first month
#of the earliest year in the proposed time period
start_date = '2013-01-01'
end_date = '2022-01-01'
stock_data = yf.download(tickers_str, start_date, end_date)

price = stock_data['Adj Close'].values
#print(price)
days = price.shape[0]
#print(days)
ret = price[1:]/price[:-1]-1 # relative return
plt.plot(ret)
plt.xlabel('time') 
plt.ylabel('') 
plt.title("Relative returns")

μ=np.mean(ret,axis=0)*250 #Annualize 250 days
Σ = np.cov(ret.T)*250 #Annualize 250 days
#print(len(μ))
#print(len(Σ))

print(f"The sample mean returns is: \n{μ} \n\nThe sample covariance matrix is: \n{Σ}\n\n")

"""###(b) Compute the **weights of the minimal-variance portfolio** and print them."""

d = len(μ) #71

#  Weight vector of the minimal-variance portfolio
α_min = np.linalg.solve(Σ,np.ones(d))
α_min = α_min/np.sum(α_min)

print(f"The weight of the minimal variance portfolio are: \n{α_min}\n\n")
plt.plot(α_min,'o')
plt.title("Weight of the minimal variance portfolio")
plt.xlabel('tickers')
plt.ylabel('α_min')

#Verification [α_1 + ... + α_d] = 1
sum = 0
for weight in α_min:
  sum += weight
#print(sum)

"""###(c) Compute the **weights of the optimal mean-variance portfolio** (i.e., maximizing a linear combination of mean and variance) with the coefficient of risk aversion γ = 2. Plot the portfolio weights on a graph. Print the mean and the standard deviation of the resulting portfolio.

"""

# Coefficient of risk aversion
γ = 2

# The function to minimized
my_fun = lambda x: -x@μ + γ*(x@(x@Σ))

# Non-linear constraint
def my_constr(x):     
    return (np.sum(x) - 1) # [α_1 + ... + α_d] - 1 = 0

constr = {'type': 'eq', #type str: Constraint type ‘eq’ for equality - Equality constraint means that the constraint function result is to be zero
          'fun': my_constr} #fun callable: The function defining the constraint

# Optimal mean-variance portfolio
opt = scipy.optimize.minimize(my_fun, α_min, constraints=constr, options={'maxiter':1e6})
#print(opt)

#  Weight vector of the optimal mean-variance portfolio
α_opt = opt.x

plt.plot(α_opt,'o')
plt.title('Weights of the optimal mean-variance portfolio')
plt.xlabel('tickers')
plt.ylabel('α_opt')
plt.show()

# Mean 
μ_opt = α_opt@μ

# Standard deviation 
std_opt = np.sqrt(α_opt@(α_opt@Σ)) #do not use the np.std because it add other errors 

print(f"For the optimal mean-variance portfolio \nThe sample mean returns is: {μ_opt}\nThe standard deviation is: {std_opt}\n")

"""RESULTS of scipy.optimize.minimize

*fun, jac: ndarray*                            
~ Values of objective function and its Jacobian

*nfev, njev : int*  
~ Number of evaluations of the objective functions and of its Jacobian

*nit: int*
~ Number of iterations performed by the optimizer

###(d) Compute the **weights of the optimal mean-variance portfolio in the robust setting**, assuming that the true mean returns of the basic assets are within one **standard deviation** (the standard deviation is estimated from the sample) away from their sample means. Plot the weights of the resulting optimal portfolio and compare this graph to the one produced in part (c). Print the standard deviation of the optimal portfolio return, as well as its worst- and best-case mean return (according to the chosen intervals of possible mean returns of the basic assets). Compare these means and the standard deviation to those produced in part (c) and explain the difference between the two results.
"""

# half size of μ interval 
ϵ = np.sqrt(np.diagonal(Σ))

# The function to minimized
my_fun_robust = lambda x: -μ@x + ϵ@abs(x) + γ*(x@(x@Σ))

# Optimal mean-variance portfolio in the robust setting
opt_robust = scipy.optimize.minimize(my_fun_robust, α_min, constraints=constr, options={'maxiter':1e6})
# print(opt_robust)

#  Weight vector of the optimal mean-variance portfolio
α_robust = opt_robust.x
plt.plot(α_robust,'o')
plt.title('Weights of the optimal mean-variance portfolio in the robust setting')
plt.xlabel('tickers')
plt.ylabel('α_robust')
plt.show()

print("The weights of the optimal mean-variance portfolio in the robust setting is closer to 0 than the same weights without the robust setting.\n")

# Mean 
μ_robust = α_robust@μ

# Standard deviation 
#std_robust = np.std((ret-μ_robust)*α_robust)
std_robust = np.sqrt(α_robust@(α_robust@Σ))

print(f"For the optimal mean-variance portfolio in a robust setting \nThe sample mean returns is: {μ_robust}\nThe standard deviation is: {std_robust}\n")

"""###(e) Compute the **efficient frontier** and plot it as a set **{(f(μ),μ)}**, where μ changes over a grid of 100 equidistant points in [0, 2], and **f(μ) is the standard deviation** of the efficient portfolio with **mean return μ**. On the same plot, show the pairs **(√Σii, μi)** corresponding to the standard deviations and the means of the returns of individual basic assets. Comment on where the latter pairs lie relative to the efficient frontier and why. """

# EFFICIENT PORTFOLIO

# Initialize an array of target returns
μ_target = np.linspace(0,2,100) #there will be 100 portfolios
R = 0.01

δ_target = [] #f(μ_target): standard deviation
#Σ_target = np.zeros(d+2) 

#Setting up covariance matrix 
Σ_target = np.zeros((73,73))
Σ_target[:71,:71] = 2*Σ
Σ_target[:71,71] = -1*np.ones(71)
Σ_target[:71,72] = -1*μ
Σ_target[71,:71] = np.ones(71)
Σ_target[72,:71] = μ

#Setting up b vector
b = np.zeros(73)
b[-2] = 1

for mu in μ_target:

  b[-1]= mu
  
  α = np.linalg.solve(Σ_target,b)[:-2]
  α = α /np.sum(α) #normalize
  
  #Find the standard deviation
  δ_target.append(np.sqrt(α.T @ Σ @ α)) #f(μ_target)

print (f'The list of f(μ) is: \n{δ_target}\n\n')

# INDIVIDUAL BASIC ASSETS

# Standard deviation
δ_ind = np.sqrt(np.diag(Σ))
#print(Σ_ind)

# Mean 
μ_ind = []
for i in range(len(tickers)):
  price_ind = stock_data['Adj Close'][tickers[i]].values
  ret_ind = price_ind[1:]/price_ind[:-1]-1 # relative return
  μ_ind.append(np.mean(ret_ind,axis=0)*250) #Annualize

#print (f'The list of μ is: \n{μ_ind}\n\nThe list of f(μ) is: \n{δ_ind}\n\n')

plt.plot(δ_target,μ_target, label='Efficient Frontier')
plt.plot(δ_ind,μ_ind,'+', label='Individual basic assets')
plt.title('Plot of the efficient Frontier & individual basic assets')
plt.xlabel('δ')
plt.ylabel('μ')
plt.legend()
plt.show()
print('Individual basic assets are below the frontier.')

"""###(f) Add a **riskless asset** to the set of available ones. Compute the **weights** of the market portfolio (i.e., of the optimal mutual fund), as well as the **mean**, **standard deviation** and **Sharpe ratio** of its return, and print them. Compute the efficient frontier for the extended market and plot it in the same coordinates and for the same values of μ as in part (e). Plot the efficient frontier from part (e) on the same graph and comment on the relationship between the two. """

# MARKET PORTFOLIO - i.e. OPTIMAL MUTUAL FUND 

# Weights such as Σ.α_M=b
b = μ - R
α_M = np.linalg.solve(Σ,b)

if (α_M.sum()!=0) & (b@α_M > 0):
    
    # Weights
    α_M = α_M/np.sum(α_M)
    α_M[0] = 1 #initialise with the risky asset
    print(f"Market portfolio exist.\n\n")
    
    # Mean
    μ_M=μ@α_M
    print(f"The mean of the Market portfolio is {μ_M}\n\n")

    # Standard deviation
    δ_M = np.sqrt(α_M@(α_M@Σ))
    print(f"The standard deviation of the Market portfolio is {δ_M}\n\n")

    # Sharp ratio
    S = (μ_M-R)/δ_M   
    print(f"The sharp ratio of the Market portfolio is {round(S,3)} ~ 1, which means that it is a good stock\n\n")

else:
    print("market portfolio does not exist")

# EFFICIENT FRONTIER FOR THE EXTENDED MARKET

# Riskless asset
R = 0.01 #risk free
δ_rl = 0 #std
α_rl = 1  #weight

#Setting up covariance matrix 
Σ_ext_M2 = np.zeros((72,72))
Σ_ext_M2[1:,1:]=Σ

Σ_ext_M = np.zeros((74,74))
Σ_ext_M[:72,:72] = 2*Σ_ext_M2
Σ_ext_M[:72,72] = -1*np.ones(72)
Σ_ext_M[0,73] = - R
Σ_ext_M[1:72,73] = -1*μ
Σ_ext_M[72,:72] = np.ones(72)
Σ_ext_M[73,0] = R
Σ_ext_M[73,1:72] = μ

#Setting up b vector
b = np.zeros(74)
b[-2] = 1

δ_ext_M  = [] #f(μ_target)

for mu in μ_target:

  b[-1]= mu
  α = np.linalg.solve(Σ_ext_M,b)[:-2] # don't need lambdas
  if (np.sum(α)!=0) and (np.sum(α)*np.sum(α*(μ_target[i]-R))):
    α = α /np.sum(α) #normalize
  else:
    print(f'Problem at iteration {i}')

  #Find the standard deviation
  δ_ext_M.append(np.sqrt(α.T @ Σ_ext_M2 @ α)) #f(μ_target)

print (f'The list of f(μ) is: \n{δ_ext_M}\n\n')

#fig,ax = plt.subplots(figsize = (15,5))
#plt.title("Efficient Frontier Extended",size=15)
plt.plot(δ_target,μ_target, label='Efficient Frontier')
plt.plot(δ_ext_M ,μ_target, label='Efficient frontier for the extended market')
plt.plot(δ_ind,μ_ind,'+', label='Individual basic assets')
plt.xlabel("δ")
plt.ylabel("μ")
plt.legend()
plt.show()
print('The efficient frontier for the extended market is tangent to the initial one.')

"""##2. The regression interpretation of CAPM.

### (a) Compute the **β** for each basic asset, according to the CAPM formula (using the part of the formula that expresses beta through the weights of the market portfolio, which you computed in 1.f), and print the results.
"""

#init with the riskless asset as my cov matrix don't have the right length compare to α_M vector
β_CAPM=((α_M@Σ)/(α_M@(α_M@Σ)))
print(f"β for each basic asset (with the riskless asset) is: \n{β_CAPM}\n\n")

"""### (b) Use the (ordinary least-square) **linear regression model**, to regress the excess returns of the **individual basic assets** on the excess return of the **market portfolio**. Recall that we denote the mean returns of the hedged assets by {ai}_(i=1,…,30) (“hedged” means that we subtract βi(R^(M) − R) from the return). Print the resulting {(ai, βi)}. Comment on the magnitude of {ai} and compare the resulting {βi} to those obtained in part (a)."""

β=[]
a=[]
p=[]
r2=[]

# Return for the Market portfolio
ret_M = ret @ α_M - R

for i in range(len(ret[0])):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(ret_M, ret[:,i])
    #   ret: excess returns of the individual basic assets
    # ret_M: excess return of the market portfolio

    β.append(slope)
    a.append((intercept-R+R*slope)*250)
    p.append(p_value)
    r2.append(r_value**2)

plt.plot(a)
plt.title('a-values')
plt.show()


plt.plot(β_CAPM,label='β-values for the CAPM formule')
plt.plot(β, label='β-values')
plt.legend()
plt.show()
print('β-values are the same \n\n')

plt.plot(p,label='p-values')
plt.plot(r2,label='R-squared')
plt.legend()
plt.show()
print(f'The p-values are low. The measure (1-p) of validity confidence is good.\n')
print(f'The R-squared are around 20%. The measure of strenght is good.\n\n')
#print(f'The resulting (a, β) are {a,β}')