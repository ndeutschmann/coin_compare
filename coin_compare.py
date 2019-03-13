import numpy as np
from numpy.random import binomial
from __future__ import division
from math import sqrt

def generate_test_measure_sample(p,N1,N2,MC_stat=1000):
    # Generate data: MC_stat repetitions of N1 coin flips and N2 coin flips
    # with probability p
    data=binomial([N1,N2],p,(MC_stat,2))/np.array([N1,N2])
    deltap = np.abs(np.subtract(data[:,0],data[:,1]))
    # For each repetition, compute the test measure
    ts = deltap/sqrt((2*p*(1-p)*2/(N1+N2)))  
    return ts

def how_many_sigma_for_alpha(p,N1,N2,alpha=0.05,MC_stat=1000):
    # Compute the number of standard deviations needed to reach
    # a significance of alpha in our test statistic (which has SD=1)
    data = generate_test_measure_sample(p,N1,N2,MC_stat)
    # Get the cumulative distribtion
    histodata = np.histogram(data,bins=10)
    histo = 1.-np.cumsum(histodata[0]/MC_stat)
    # Get the first bin in the cumulative histogram above the limit p-value
    limit_bin = np.min(np.where(histo<alpha))
    # Get the associated number of sigmas by looking at the bin limits:
    return 0.5 * (histodata[1][limit_bin] + histodata[1][limit_bin+1])

def coin_compare(N1,p1,N2,p2,alpha=0.05,MC_stat=1000):
    # Compute the test measure
    p = (N1*p1 + N2*p2)/(N1+N2)
    t = abs((p1-p2)/sqrt((2*p*(1-p)*2/(N1+N2))))
    # Compute the likelihood of the test measure
    nsig = how_many_sigma_for_alpha(p,N1,N2,alpha,MC_stat)
    return t>nsig
