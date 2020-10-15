# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:12:05 2019

@author: christian breinholt
"""




import torch
import pyro
import math

from torch.distributions import constraints

from torch.distributions.utils import broadcast_all

#import bvm_class_unimodal_NoValErr as bvm_unimodal
import bivariate_von_mises_sine_model_unimodal_15_10_2020 as bvm_unimodal



class Sine_Skewed_Bivariate_Von_Mises_Sine_model(bvm_unimodal.BivariateVonMises):
    """
    Sine skewed Bivariate von Mises distribution on the torus
    
    This distribution is a child class of Bivariate Von Mises (unimodal) Sine model
    
    Modality:
        If lam^2 / (k1*k2) > 1, the distribution is bimodal, otherwise unimodal.
            - This distribution is only defined for some 'slightly' bimodal cases (alpha < -7)
    
    :param torch.Tensor L1, L2: skew-factors.
        - L1 & L2 has to be real numbers and comply with: |L1| + |L2| <= 1.0

    
    :param torch.Tensor mu, nu: an angle in radians.
        - mu & nu can be any real number but are interpreted as 2*pi

    :param torch.Tensor k1, k2: concentration parameter
        - This distribution is only defined for k1, k2 > 0

    :param torch.Tensor lam: correlation parameter
        - Can be any real number, but is not defined for very bimodal cases
        - See 'Modality' above
        
    :param torch.Tensor w: reparameterization parameter
        - Has to be between -1 and 1
    
    
    This distribution is build from the article 'Sine-skewed toroidal distributions 
        and their application in protein bioinformatics'
    Authors of the article: Jose Ameijeiras-Alonso and  Christophe Ley, from KU leuven and Ghent University.
    
    This distribution was written by Christian Sigvald Breinholt from Copenhagen University, 
        under supervision of Associate Professor Thomas Wim Hamelryck.
    """
    arg_constraints = {'L1': constraints.real, 'L2': constraints.real} 
    ## The other parameter constraints are in the parent class
    
    
    support = constraints.real
    has_rsample = False
        
    
    def __init__(self, L1, L2, mu, nu, k1, k2, lam=None, w=None, validate_args=None):
        
        ## It's important that |L1| + |L2| is <= 1 and that both are on the range -1 to 1. For now the check has been disabled as it cause issues with enumration
        """
        if all(  ((torch.abs(L1) + torch.abs(L2)) > torch.ones((len(L1),1)))) == True:
        
            raise ValueError("|skew1| + |skew2| has to be less than 1")"""
        
        
        self.L1, self.L2 = broadcast_all(L1, L2)
    
        super().__init__(mu = mu, nu = nu, k1 = k1, k2 = k2, lam = lam, w = w)
        
    
    def log_prob(self, phi_psi): ## 
        # Actual likelihood function
        """ log Joint distribution of phi and psi """
        phi = phi_psi[:,0]
        psi = phi_psi[:,1]
        
        
        if len(phi) > int(1):
            phi = phi.reshape(len(phi), 1)
            psi = psi.reshape(len(psi), 1)
        
        
        return( (super().log_prob(phi_psi) ) + torch.log( (1 + self.L1 * torch.sin(phi - self.mu) + self.L2 * torch.sin( psi - self.nu ))))
        
        
    def sample(self, sample_shape=torch.Size([])):
        # 'Sine-skewed toroidal distributions and their application in protein bioinformatics'
        # Authors of the article: Jose Ameijeiras-Alonso and  Christophe Ley, from KU leuven and Ghent University.
        # 
        """
        marg: marginal distribution (using _acg_bound())
        cond: conditional distribution using a modified univariate von Mises
            - as described in Singh et al. (2002)
        """
        
        if sample_shape == torch.Size([]):
            sample_shape = 1
        Y = super().sample(sample_shape)
        
        Y1_list = torch.empty((sample_shape))
        Y2_list = torch.empty((sample_shape))
        
        for i in range(sample_shape):
            Y1 = Y[i,0]
            Y2 = Y[i,1]
            
            U = pyro.sample("U", pyro.distributions.Uniform(0., 1.))
            
            # Y1:
            if U <= ( ( 1 + self.L1 * torch.sin(Y1 - self.mu) + self.L2 * torch.sin(Y1 - self.nu ) ) / 2):
                Y1_new = Y1
            else:
                Y1_new = -Y1 + 2*self.mu
            
            #Y2:
            if U <= ( ( 1 + self.L1 * torch.sin(Y2 - self.mu) + self.L2 * torch.sin(Y2 - self.nu ) ) / 2):
                Y2_new = Y2
            else:
                Y2_new = -Y2 + 2*self.nu
            
            Y1_list[i] = Y1_new
            Y2_list[i] = Y2_new

            
        
        return (torch.stack( (Y1_list, Y2_list) )).T

    
    
    def expand(self, batch_shape):
        
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            mu = self.mu.expand(batch_shape)
            nu = self.nu.expand(batch_shape)
            k1 = self.k1.expand(batch_shape)
            k2 = self.k2.expand(batch_shape)
            lam = self.lam.expand(batch_shape)
            L1 = self.L1.expand(batch_shape)
            L2 = self.L2.expand(batch_shape)
            return type(self)(mu, nu, k1, k2, lam, L1, L2, validate_args=validate_args)
            
        



