# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.integrate as spint
import scipy.optimize as sopt
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        return self.bsm_model.impvol(self.price, strike, spot, texp=texp, cp=1)
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1, random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        N=120
        n_path=100
        #####
        vo_paths=[]
        mt_vol=np.linspace(self.vov,self.vov,N)  #treat N discreate time as N asset

        sigma = pf.BsmNdMc(mt_vol, cor=0,rn_seed=random_seed)
        sigma.simulate(tobs = [texp/N], n_path=n_path)
        sigma_paths=list(sigma.path[0].transpose())

        #generate 100 sigma paths
        vo_paths.append(list(sigma_paths[0]* self.sigma))
        for i in range(1,120):
        	vo_paths.append(list(sigma_paths[i]* vo_paths[i-1]))

        random_z=sigma._bm_incr(tobs = [texp/N], n_path=n_path)[0].transpose()


        #generate 100*100 stock price paths,correlated with Z
        np.random.seed(random_seed)
        z2=np.random.normal(size=(N,n_path))*self.vov*np.sqrt(texp/N)
        w=(self.rho*random_z+np.sqrt(1-self.rho)*z2)/(self.vov*np.sqrt(texp/N))

        s1_paths=spot*np.exp(-0.5*texp/N*self.sigma**2+self.sigma*np.sqrt(texp/N)*w[0])
        price_path=np.array([np.tile(s1_paths,n_path)])
        for j in range(1,120):
        	vo_path=np.array(vo_paths[j])
        	sn_path=np.array([])
        	multi=np.exp(-0.5 * texp/N * vo_path[:,None]**2 + np.sqrt(texp/N) * vo_path[:,None] * w[j])
        	sn_path=multi.reshape(-1)
        	price_path=np.append(price_path,[sn_path*price_path[j-1]],axis=0)

        return np.mean(np.fmax(price_path[-1]-strike[:,None],0),axis=1)

        

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        return self.normal_model.impvol(self.price, strike, spot, texp=texp, cp=1)
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1, random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        N=120
        n_path=100
        #####
        vo_paths=[]
        mt_vol=np.linspace(self.vov,self.vov,N)  #treat N discreate time as N assets

        sigma = pf.BsmNdMc(mt_vol, cor=0,rn_seed=random_seed)
        sigma.simulate(tobs = [texp/N], n_path=n_path)
        sigma_paths=list(sigma.path[0].transpose())

        #generate 100 sigma paths
        vo_paths.append(list(sigma_paths[0]* self.sigma))
        for i in range(1,120):
        	vo_paths.append(list(sigma_paths[i]* vo_paths[i-1]))

        random_z=sigma._bm_incr(tobs = [texp/N], n_path=n_path)[0].transpose()


        #generate 100*100 stock price paths,correlated with Z
        np.random.seed(random_seed)
        z2=np.random.normal(size=(N,n_path))*self.vov*np.sqrt(texp/N)
        w=(self.rho*random_z+np.sqrt(1-self.rho)*z2)/(self.vov*np.sqrt(texp/N))

        s1_paths=spot + self.sigma*np.sqrt(texp/N)*w[0]
        price_path=np.array([np.tile(s1_paths,n_path)])

        for j in range(1,120):
        	vo_path=np.array(vo_paths[j])
        	sn_path=np.array([])
        	multi= np.sqrt(texp/N) * vo_path[:,None] * w[j]
        	sn_path=multi.reshape(-1)
        	price_path=np.append(price_path,[price_path[j-1]+sn_path],axis=0)

        return np.mean(np.fmax(price_path[-1]-strike[:,None],0),axis=1)

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return self.bsm_model.impvol(self.price, strike, spot, texp=texp, cp=1)

    
    def price(self, strike, spot, texp=None, cp=1, random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        n_path=1000
        N=120

        #generate 1000 sigma paths
        mt_vol=np.linspace(self.vov,self.vov,N)  #treat N discreate time as N asset

        m = pf.BsmNdMc(mt_vol, cor=0,rn_seed=random_seed)
        m.simulate(tobs = [texp/N], n_path=n_path)
        sigma_path=m.path[0].transpose()

        #generate 100 sigma paths
        vo_paths=np.array([self.sigma*sigma_path[0]])
        for i in range(1,120):
        	vo_paths=np.append(vo_paths,[sigma_path[i] * vo_paths[i-1]],axis=0)        


        vo_final = vo_paths[-1]
        it = spint.simps(vo_paths**2, dx=texp/N,axis=0)/(self.sigma**2*texp)

        spot_equiv = spot * np.exp((self.rho/self.vov)*(vo_final-self.sigma) - 0.5*self.rho**2*self.sigma**2*texp*it)
        sigma_bs_equi = self.sigma*np.sqrt((1-self.rho**2)*it)

        bs_m = pf.Bsm(sigma_bs_equi, intr=self.intr, divr=self.divr)
        price=np.mean(bs_m.price(strike=strike[:,None],spot=spot_equiv,texp=texp,cp=1),axis=1)

        return price

'''
Conditional MC model class for Beta=0
'''

class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        return self.normal_model.impvol(self.price, strike, spot, texp=texp, cp=1)

    def price(self, strike, spot, texp=None, cp=1, random_seed=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        n_path=1000
        N=120

        #generate 1000 sigma paths
        mt_vol=np.linspace(self.vov,self.vov,N)  #treat N discreate time as N asset

        m = pf.BsmNdMc(mt_vol, cor=0,rn_seed=random_seed)
        m.simulate(tobs = [texp/N], n_path=n_path)
        sigma_path=m.path[0].transpose()

        vo_paths=np.array([self.sigma*sigma_path[0]])
        for i in range(1,120):
        	vo_paths=np.append(vo_paths,[sigma_path[i] * vo_paths[i-1]],axis=0)        

        vo_final = vo_paths[-1]
        it = spint.simps(vo_paths**2, dx=texp/N,axis=0)/(self.sigma**2*texp)

        spot_equiv = spot + (self.rho/self.vov)*(vo_final-self.sigma) 
        sigma_norm_equi = self.sigma*np.sqrt((1-self.rho**2)*it)

        norm_m = pf.Norm(sigma_norm_equi, intr=self.intr, divr=self.divr)
        price = np.mean(norm_m.price(strike=strike[:,None],spot=spot_equiv,texp=texp,cp=1),axis=1)      

        return price
