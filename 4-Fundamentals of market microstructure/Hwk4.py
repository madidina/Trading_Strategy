#!/usr/bin/env python
# coding: utf-8

# # Math 584 - Homework 4
# *Diane PERES*
# 
# ---

# In[1]:


import numpy as np
import os
import scipy.io as sio  # for loading matlab data
from scipy import stats
from statsmodels.tsa import stattools
from statsmodels.regression import linear_model
import warnings
warnings.filterwarnings('ignore')# get warning for complex type
import time 


# ## MSFT

# In[2]:


#data = sio.loadmat('./tickers/MSFT_20141103.mat')
dataDir = "./tickers/"
mats_MSFT = []
indices = ["03","04","05","06","07","10","11","12","13","14","17","18","19","20","21","24","25","26"]
for i in indices:
    mats_MSFT.append( sio.loadmat( dataDir+"MSFT_201411"+i+".mat" ) )
    #print(dataDir+"MSFT_2014110"+i+".mat" )


# ## GOOG

# In[3]:


#data = sio.loadmat('./tickers/MSFT_20141103.mat')
dataDir = "./tickers/"
mats_GOOG = []
indices = ["03","04","05","06","07","10","11","12","13","14","17","18","19","20","21","24","25","26"]
for i in indices:
    mats_GOOG.append( sio.loadmat( dataDir+"GOOG_201411"+i+".mat" ) )
    #print(dataDir+"GOOG_2014110"+i+".mat")


# # Exercice 1
# In this exercise, you will estimate the coefficient of a linear price impact function and will deduce the risk-aversion of the (hypothetical) market maker from it.

# ## (a) 
# (8 pts) Estimate the price impact coefficient λ, as discussed in Section 5 of Chapter 4, using the model:
# 
# ∆S(t) = λ∆V(t) + ση(t)
# 
# * V is the **signed traded volume**, 
# * S is the midprice, 
# * η is a zero-mean unit-variance white noise,  
# * ∆St = S(t+∆t) − St
# * ∆Vt = V(t+∆t) − Vt
# 
# Use a reduced-frequency sample of (V,P), by choosing **∆t that corresponds to one second** (i.e., you consider the values of (S,V) at the beginning of every second). Use the least-squares linear regression to estimate λ for each of the tickers MSFT and GOOG, using the data from Nov 3 through Nov 26, 2014, as a sample for each ticker.

# ### Notes 
# On parcours 1 fichier par jours. Chaque jour il y a t= 234 000 times. On doit travailler avec *∆t = 1 s = 1e3 ms*, donc repartir la prise de valeur toutes les 1 secondes. Iteration / loop each days every 1 seconds 
# * t = tableau des temps, each loop there is 234000 miliseconds
# * Le reste des autres tableau fait la même taille
# 
# λ is actually the slope of the linear price impact function

# In[4]:


def lambda_LS (mats):
    dS, dV = [], []
    dt = 10 # increments every 10*0.1 = 1 seconds

    # MSFT
    for day in range(len(mats)):
        LOB = mats[day]['LOB']
        bid = np.array(LOB['BuyPrice'][0][0]*1e-4)
        ask = np.array(LOB['SellPrice'][0][0]*1e-4)
        vMO = np.array(LOB['VolumeMO'][0][0]*1.0)#total volume of MOs submitted in the given 0.1 second interval [ask,bid]

        V   = list(np.cumsum(vMO[:,0]-vMO[:,1])[::dt])
        dV += list(np.diff(V))

        S   = list((ask[:,0]+bid[:,0])*0.5)[::dt]
        dS += list(np.diff(S))

    # least-squares linear regression 
    # slope is lambda 
    slope_a, intercept, r, p, std_err = stats.linregress(dV, dS)
    print(f'λ = {slope_a}')
    print(f'p = {p}')
    print(f'r2 = {r**2}')
    return(slope_a)

print('MSFT')
slope_a_MSFT = lambda_LS(mats_MSFT)

print('\nGOOG')
slope_a_GOOG = lambda_LS(mats_GOOG)


# ## (b) Limit Order Book
# Estimate the price impact coefficient λ via **LOB**. At each time *t*, consider the volumes *{V^(a,i)_t, V^(b,i)_t}* (for i = 1 to 10) on the ask and bid sides of the LOB, respectively, at the first 10 price levels {P^(a,i)_t, P^(b,i)_t}10 (for i = 1 to 10) (with corresponding to the best price). Imagine a flat LOB with the same total volume (over the first 10 levels of LOs) on each side. 
# The **height** of the latter (imaginary) LOB on
# * the bid side is H^b_t 
# * the ask side is H^a_t 
# * λ_t can be viewed as the (approximate) instantaneous price impact at time t 
# 
# Estimate the average price impact **λ as the sample mean of {λt}** for each of the tickers MSFT and GOOG, using the data from Nov 3 through Nov 26, 2014, as a sample for each ticker.
# 
# Compare the resulting estimates of λ, for each ticker, to those obtained in part (a) and explain the difference. 
# 
# Which estimation method would you use when designing a trading strategy and why?

# In[5]:


def lambda_LOB(mats):
    slope = []
    dt    = 10

    # MSFT
    for day in range(len(mats)):
        LOB    = mats[day]['LOB']
        bid    = np.array(LOB['BuyPrice'][0][0]*1e-4)
        bidvol = np.array(LOB['BuyVolume'][0][0]*1.0)
        ask    = np.array(LOB['SellPrice'][0][0]*1e-4)
        askvol = np.array(LOB['SellVolume'][0][0]*1.0)

        S      = (ask[:,0]+bid[:,0])*0.5
        Ha, Hb = np.sum(askvol[:,:dt],axis=1)/(ask[:,9] - S), np.sum(bidvol[:,:dt],axis=1)/(S - bid[:,9])
        slope += list(0.5 * ( 1/Ha + 1/Hb ) )

        #for t in range(len(T)):

            #S = (ask[t][0] + bid[t][0])/2
            #Pa, Pb = ask[t][9], bid[t][9]
            #Va, Vb = sum(askvol[t]), sum(bidvol[t])
            #ha, hb = 1/(Pa - S) * Va, 1/(S - Pb) * Vb
            #s.append(0.5 * ( 1/Ha + 1/Hb ) )

    slope_b = np.mean(slope)
    print(f'λ = {slope_b}')
    return(slope_b)

print('MSFT')
slope_b_MSFT = lambda_LOB(mats_MSFT)

print('\nGOOG')
slope_b_GOOG = lambda_LOB(mats_GOOG)


# ### The best strategy is the one with the **smalest λ** !

# ## (c)
# Recall that, in the utility indifference model (or in Grossman-Miller model with one market maker), the price impact coefficient is given by
# 
# λ = 2γσ2,
# 
# * **γ** measures the **risk aversion** of the market maker (the higher is γ, the more “afraid” the market maker is of the risk)
# * **σ2** is the **variance** of the “fundamental value” of the asset
# 
# The goal of parts (c) and (d) is to **estimate γ**. 
# 
# While it is hard to measure the fundamental value of the asset, we can pretend that it is given by the midprice of the stock at the end of the next 5-minute time interval. Thus, to approximate σ2, we denote by **h** the number of time stamps in a 5-minute time interval (for the sample data we have), and consider the associated 5-minute increments of the midprice, S_(i+1)h − S_ih. **We approximate σ2 by the sample variance of {S_(i+1)h − S_ih}**.
# 
# Estimate σ2 for each of the tickers MSFT and GOOG, using the data from Nov 3 through Nov 26, 2014, as a sample for each ticker. Compare the estimated values of σ2 for each ticker. What does this comparison tell you about the behavior of the prices of these two stocks?
# 
# ## (d)
# Use the value of σ2, estimated in part (c), and the two estimates of λ, obtained in parts (a) and (b), to produce four estimates of γ via *λ = 2γσ2*. 
# Comment on how these estimates compare to each other, across the two tickers and across the two estimation methods.
# 
# 
# *Note : 5 min = 300 seconds = 300e3 miliseconds*

# In[6]:


def risk_aversion_variance(mats, slope_a, slope_b):
    dS = []
    h = int(10*60*5) #the number of time stamps in a 5-minute time interval with 0.1 seconds incrementation

    for day in range(len(mats)):

        LOB = mats[day]['LOB']
        T   = (np.array((LOB['EventTime'][0][0][:,0]))*1e-3)#time in seconds, measured from NASDAQ opening at 9:30am
        bid = np.array(LOB['BuyPrice'][0][0]*1e-4)
        ask = np.array(LOB['SellPrice'][0][0]*1e-4)
        S   = []
        S   = list(((ask[:,0]+bid[:,0])*0.5)[::h])
        dS += list(np.diff(S))

        #for i in range(int(len(T)/h-1)):
            #S+=list((ask[(i*h):((i+1)*h),0]+bid[(i*h):((i+1)*h),0])*0.5)

    sigma = (np.std(dS))**2
    print('σ2 = ',sigma)

    # a)
    gamma_A = slope_a / (2* sigma)
    print(f'a) First methode, γ = {gamma_A}')

    # b)
    gamma_B = slope_b / (2*sigma)
    print(f'b) Second methode, γ = {gamma_B}')

print('MSFT')
risk_aversion_variance(mats_MSFT, slope_a_MSFT, slope_b_MSFT)

print('\nGOOG')
risk_aversion_variance(mats_GOOG, slope_a_GOOG, slope_b_GOOG)


# # Exercice 2.
# In this exercise, you implement and test the trading strategy described in Section 5.1 of Chapter 4, using the data for MSFT ticker on Nov 3, 2014, as your sample. Compute the **order flow OF** using only the **LO volumes at the 10 best price levels**. Use DF test of order one, to test the order flow for stationarity. The parameters’ values are as follows (using the notation of Chapter 4)
# 
# * q = 100
# * Trading time range: 10am–3:30pm,
# * Length of estimation window: 1 minute,
# * Length of prediction interval: 10 seconds,
# * tP∗ = 0.00999, lP∗ = 0.00999, p∗_1 = 0.01, p∗_2 = 0.05, α∗ = 0.005
# 
# (The reason for the above choice of tP∗ and lP∗, which are not given by integer numbers of cents, is to protect the algorithm against the numerical (rounding) errors, which arise when the data is transferred from a MatLab format to Python.)
# 
# Compute the PnL of this strategy for every time stamp in the testing range 10:01am–3:30pm, so that, at any time t, you define the associated increment of the PnL process as the profit/loss from a roundtrip trade if this trade is terminated (i.e., the position is closed) at time t, and you set this increment to zero otherwise. **Plot the resulting PnL process and compute the Sharpe ratio of its absolute returns**. Annualize the Sharpe
# ratio by multiplying it by
# √(250·n), where n is the length of the PnL process.

# ### Notes:
# 1e3 ms = 1 s 
# 
# Time slope open at 34200 s = 9:30am and close at 57600 s = 16pm
# 
# We want to focus on the trading range 10am – 3:30pm = 60* 60* 10- 60* 60* 15.5
# 
# So we will look 30min after the begining and 30min before the end.

# In[70]:


q = 100 # nbr shares
t_gap = int(10*60*31) # 31min 
end = int(10*60*6*60)
N = int(10*60) # length of estimation window 1 min 
T = int(10*10) # Length of prediction interval 10 seconds
tP = 0.00999 # target price change
lP = 0.00999 #stop loss price change
p_1, p_2 = 0.01, 0.05 # largest admissible p-value for DF test and for the price impact regression
alpha_star = 0.005 #minimum required “strength”


# In[71]:


# Measure are made every 0.1 seconds     
LOB    = mats_MSFT[0]['LOB']
Time   = (np.array((LOB['EventTime'][0][0][:,0]))*1e-3)[t_gap:-t_gap:]#time in seconds, measured from NASDAQ from 10h30 to 15h30
bid    = np.array(LOB['BuyPrice'][0][0]*1e-4)[t_gap:-t_gap:]
ask    = np.array(LOB['SellPrice'][0][0]*1e-4)[t_gap:-t_gap:]
bidvol = np.array(LOB['BuyVolume'][0][0]*1.0)[t_gap:-t_gap:]
askvol = np.array(LOB['SellVolume'][0][0]*1.0)[t_gap:-t_gap:]
size   =len(Time)

S   = list((ask[:,0]+bid[:,0])*0.5)
OF  = list(np.sum(bidvol[:,:10],axis=1)-np.sum(askvol[:,:10],axis=1))

#print('size = ',size)
#print(len(S), len(OF))


# In[72]:


start_time = time.time()
PnL = []
pos = ['close',0] #statue of the position : nothing/long/short and time when the position was open

for t in range(N, size):
    pnl = 0
    if pos[0] == 'close': # we have no position at time t

        # i) reduce the sample of midprice S and order flow F over the trading window [t−N,t] every 10 values
        OF_centered = (OF[t-N:t:10]-np.mean(OF[t-N:t:10]))
        dOF         = list(np.diff(OF[t-N:t:10]))
        dS          = list(np.diff(S[t-N:t:10]))
        
        # ii) Fit AR model to F

        # DF
        my_DF = stattools.adfuller(OF_centered,1,'c',None)
        p1 = my_DF[1]
        phi, sigma = linear_model.yule_walker(OF_centered,order=1)

        # LS
        slope, intercept, r, p2, stderr = stats.linregress(dOF, dS)
        res = [dS[i] - slope*dOF[i] - intercept for i in range(len(dS))]
        sigma_hat = np.std(res)
        """
        print('t    = ',t)
        print('p1   = ',p1)
        print('phi  = ',phi[0])
        print('σ    = ', sigma)
        print('p2   = ',p2)
        print('λ    = ',slope,'\n')
        """
        # iii) 
        if (slope > 0) and (p1 <= p_1 ) and ( p2 <= p_2 ):
            alpha = - slope * OF_centered[-1] * (1 - phi[0])
            """
            print('α    = ', alpha)
            print('α/σ  = ',alpha/(sigma_hat**0.5),'\n')
            """
            if (alpha/(sigma_hat)) > alpha_star:
                #print('open a long position at t = ',t,'\n')
                pos = ['long',t]
                pnl = q*ask[t,0]
            
            elif (alpha/(sigma_hat)) < - alpha_star:
                #print('open a short position at t = ',t,'\n')
                pos = ['short',t]
                pnl = q*bid[t,0]
                
    elif pos[0] == 'long':
        if (bid[t,0]-bid[pos[1],0] >= tP) or (ask[t,0]-ask[pos[1],0] <= -lP) or (t - pos[1] > T):
            #closing our position via LOs by selling our shares at price At
            #print('close a long position at t = ',t,'\n')
            pnl = ask[pos[1],0]-ask[t,0]
            pos = ['close',t]
            
    elif pos[0] == 'short':
        if (ask[t,0]-ask[pos[1],0] <= -tP) or (bid[t,0]-bid[pos[1],0] >= lP) or (t - pos[1] > T):
            #closing our position via LOs by buying our shares at price Bt
            #print('close a short position at t = ',t,'\n')
            pnl = bid[t,0]-bid[pos[1],0]
            pos = ['close',t]
    PnL.append(pnl)
    
print(f'---{time.time() - start_time} seconds---')

n     = len(PnL)
sharp = np.mean(PnL)/np.std(PnL) * (n*250)**0.5
print(n) 
print(sharp)


# My sharp ratio is too big !! 
