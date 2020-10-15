# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:17:40 2020

@author: christian Sigvald Breinholt
"""

import argparse
import logging
import sys



import torch
import torch.nn as nn
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta, init_to_uniform, init_to_sample
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO

from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings



########################################################################################################
# Custom:
from generate_and_plotting_full_model import plot_rama, generate_sequence, plot_data_provide_clusters, plot_ELBO, plot_compare, plot_compare_histograms, generate_dihedral_sequence_BIVARIATE, generate_sequence_DSSP_AA
import dataloader_top500_sequences_D1 as dataloader_top500_sequences
#import bvm_class_unimodal_NoValErr as bvm
import bivariate_von_mises_sine_model_unimodal_15_10_2020 as bvm

##########################################################################################################

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
#
#
# When you run this file it will generate a lot of plots which will be saved in the same directory as this file is in
#
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

    
def model_1(data_Dihedral, data_DSSP, data_AA, data_lengths, priors_mu_nu, args, batch_size=None, include_prior=True):
    """ This is TorusDBN using the Bivariate Von Mises Sine model
    This is an alteration of HMM.py example from Pyros examples
    https://github.com/pyro-ppl/pyro/blob/dev/examples/hmm.py
    or
    https://pyro.ai/examples/hmm.html
    You should take a look at it, as it is very well commented and will make the following code easier to understand"""
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, data_Dihedral.shape)
        assert data_lengths.shape == (num_sequences,)
        assert data_lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
        
        #######################################################
        # probs_x is the transition matrix
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(torch.ones(args.hidden_dim, args.hidden_dim) * 10000.)
                                  .to_event(1))
        
        ######################################################
        # the next 7 probs_* can be thought of as emission matrices
        probs_mu = pyro.sample("probs_mu",
                              dist.VonMises(torch.ones(args.hidden_dim)*priors_mu_nu[:,0], torch.ones(args.hidden_dim)*300.)
                                  .to_event(1))
        
        probs_nu = pyro.sample("probs_nu",
                              dist.VonMises(torch.ones(args.hidden_dim)*priors_mu_nu[:,1], torch.ones(args.hidden_dim)*300.)
                                  .to_event(1))
        
        probs_kappa1 = pyro.sample("probs_kappa1",
                              dist.Uniform(torch.ones(args.hidden_dim, dtype=torch.float)*5., torch.ones(args.hidden_dim, dtype=torch.float)*300.)
                                  .to_event(1))
        probs_kappa2 = pyro.sample("probs_kappa2",
                              dist.Uniform(torch.ones(args.hidden_dim, dtype=torch.float)*5., torch.ones(args.hidden_dim, dtype=torch.float)*300.)
                                  .to_event(1))
        probs_lam = pyro.sample("probs_lam",
                              dist.Uniform(torch.ones(args.hidden_dim, dtype=torch.float)*0., torch.ones(args.hidden_dim, dtype=torch.float)*40.)
                                  .to_event(1))
        
        probs_AA = pyro.sample( "probs_AA", dist.Dirichlet( torch.ones( args.hidden_dim, 20  ) ).to_event( 1 ) )
        
        probs_DSSP = pyro.sample( "probs_DSSP", dist.Dirichlet( torch.ones( args.hidden_dim, 3 ) ).to_event( 1 ) )
        
        
       
        
    # 3 plates are needed, 1 for the conditional independence of sequences given what is outside the plate.
    # The reason for 2 emission plates is that the size of the observations is 2 for dihedral angles but only 1 for the DSSP and AA labels.
    # in the TorusDBN_Tandem_Von_Mises, you will see that only 1 emission plate is used. This is because when using two von mises instead of the bivariate von mises, each dihedral angle is size 1, just as the labels.
    emissions_plate2 = pyro.plate("emissions2", 2, dim=-1)
    emissions_plate1 = pyro.plate("emissions1", 1, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
    
        lengths = data_lengths[batch].long() #This is used to determine the length of the markov chain (how many time steps)
        x = 0 #x is the state we are currently in. This gets updated for each time step.

        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)): #mask is used for omitted data. the data has been padded with zeros and we do not wish to include the padded parts of the sequences so they are omitted from computation
                
                # we now sample the next hidden state given the previous hidden state.... markov process
                # Look up enumeration: https://pyro.ai/examples/enumeration.html
                # In short, enumeration is a way of marginalizing out DISCRETE latent variables. Here being the hidden states. This can only be used for discrete cases, meaning you're gonna ahve a bad time trying to use it with a continious time model
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                
                
                with emissions_plate1:
                    
                    # it's important to copy the observations as done here, before using them in the obs statement. it appers to be a bug when using enumeration, but it's very important otherwise you will run into problems
                    #https://forum.pyro.ai/t/categorical-and-enumeration/652/5
                    
                    # Vindex is an advanced indexing technique needed because paralell enumeration is used in this model.
                    
                    obs_DSSP = (Vindex(data_DSSP)[batch, t])#.unsqueeze(-1)
                    obs_AA = (Vindex(data_AA)[batch, t])#.unsqueeze(-1)
                    
                    pyro.sample("y_DSSP_{}".format(t), dist.Categorical(probs_DSSP[x]),
                                obs=obs_DSSP)
                    
                    pyro.sample("y_AA_{}".format(t), dist.Categorical(probs_AA[x]),
                                obs=obs_AA)
                
                
                
                with emissions_plate2:
                    obs_dihedral = (Vindex(data_Dihedral)[batch, t, :]).unsqueeze(-1)
                    
                    pyro.sample("y_dihedral_{}".format(t), bvm.BivariateVonMises(mu = probs_mu[x], 
                                                                            nu = probs_nu[x], 
                                                                            k1 = probs_kappa1[x], 
                                                                            k2 = probs_kappa2[x], 
                                                                            lam = probs_lam[x]),
                                                                            obs=obs_dihedral)
                    




models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}




def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = dataloader_top500_sequences.data_dictionary()

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequence_lengths'])))
    
    
    
    """ setting up data """

    dic = dataloader_top500_sequences.data_dictionary()
    
    train_dic = dic['train']
    train_AA = train_dic['sequences_AA']
    train_DSSP = train_dic['sequences_DSSP']
    
    lengths = train_dic['sequence_lengths']
    sequences = train_dic['sequences_Dihedral']
    
    
    # This will create a plot and provide the priors used for the dihedral angles in model_1
    priors_mu_nu = plot_data_provide_clusters(mask = lengths.int(), AA_train = train_AA, DSSP_train = train_DSSP,nr_clusters = args.nr_clusters, hidden_dims = args.hidden_dim)

    
    test_dic = dic['test']
    test_AA = test_dic['sequences_AA']
    test_DSSP = test_dic['sequences_DSSP']

    test_Dihedral = test_dic['sequences_Dihedral']
    test_mask = test_dic['sequence_lengths']
    
    
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)
    
    
    # AutoDelta makes MAP estimates for the parameters. Think of it as Bayesian maximum likelyhood.
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")), init_loc_fn = init_to_sample)


    if args.print_shapes:
        first_available_dim = -3 # This tells Pyro the first available dim for enumeration
        guide_trace = poutine.trace(guide).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info(model_trace.format_shapes())


    optim = Adam({'lr': args.learning_rate})

    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting= 2,
                strict_enumeration_warning=True,
                jit_options={"time_compilation": args.time_compilation})
    svi = SVI(model, guide, optim, elbo)







    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    test_elbo = []
    train_elbo = []
    
    
    for step in range(args.num_steps):
        
        loss = svi.step(data_DSSP = train_DSSP, data_AA = train_AA, data_Dihedral = sequences, data_lengths = lengths, priors_mu_nu = priors_mu_nu, args=args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))
        
        if step%10 == 0:
            
            
            fakeseq_Dihedral = generate_dihedral_sequence_BIVARIATE(probs_mean1 = pyro.param('AutoDelta.probs_mu'), 
                                                probs_mean2 = pyro.param('AutoDelta.probs_nu'), 
                                                probs_x = pyro.param('AutoDelta.probs_x'),
                                                probs_k1 = pyro.param('AutoDelta.probs_kappa1'),
                                                probs_k2 = pyro.param('AutoDelta.probs_kappa2'),
                                                probs_lam = pyro.param('AutoDelta.probs_lam'))
    
            fakeseq_AA, fakeseq_DSSP = generate_sequence_DSSP_AA(probs_x_param = pyro.param('AutoDelta.probs_x'),
                                                                            DSSP_param = pyro.param('AutoDelta.probs_DSSP'),
                                                                            AA_param = pyro.param('AutoDelta.probs_AA'))
    
    
            
            
            
            
            #creates plots saves in nsame directory as this file is in
            plot_rama(fakeseq_Dihedral, plot_title = 'Ramachandran plot', 
                      subtitle = 'Data: Generated 20.000 samples given trained model \n Distribution: Bivariate Von Mises Sine model with kappa estimation and informative priors', 
                      textstr = 'Hidden states: {}, batch size: {}, nr. steps: {}/{}'.format(args.hidden_dim, args.batch_size, step+1, args.num_steps),
                      save_title = 'img5_step{}'.format(step+1))
            #creates plots and saves in nsame directory as this file is in
            plot_compare_histograms(mask = lengths.int(), AA_train = train_AA, DSSP_train = train_DSSP, 
                 AA_sequence = fakeseq_AA, DSSP_sequence = fakeseq_DSSP, 
                 Dihedral_sequence = fakeseq_Dihedral, nr_clusters = args.nr_clusters, 
                 hidden_dims = args.hidden_dim, batch_size = args.batch_size,
                 current_step = step+1, total_step = args.num_steps,
                 savetitle_end = 'comp_step_{}'.format(step+1))
            
            
                
            
            
            
        if step % args.test_freq == 0:
            # report test diagnostics
            num_observations_test = float(test_mask.sum())
            test_loss = elbo.loss(model, guide, data_DSSP = test_DSSP, 
                                  data_AA = test_AA, data_Dihedral = test_Dihedral, 
                                  data_lengths = test_mask, priors_mu_nu = priors_mu_nu, 
                                  args=args, include_prior=False)
            test_elbo.append(test_loss/num_observations_test)
            
            train_loss = elbo.loss(model, guide, data_DSSP = train_DSSP, data_AA = train_AA, data_Dihedral = sequences, data_lengths = lengths, priors_mu_nu = priors_mu_nu, args=args, include_prior=False)
            train_elbo.append(train_loss/num_observations)
            #print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            
            if step%(args.test_freq*10) == 0:
                #creates plots saves in same directory as this file is in
                plot_ELBO(total_steps = step+1, 
                          test_loss = test_elbo, 
                          train_loss = train_elbo,
                          test_frequency = args.test_freq, 
                          hidden_dims = args.hidden_dim, 
                          batch_size = args.batch_size, 
                          savetitle_end = 'intermediate_step{}'.format(step))
      
        #creates plots saves in same directory as this file is in
    plot_ELBO(total_steps = args.num_steps, 
              test_loss = test_elbo,
              train_loss = train_elbo,
              test_frequency = args.test_freq, hidden_dims = args.hidden_dim, batch_size = args.batch_size)

                  
            
    
    if args.jit and args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, data_DSSP = train_DSSP, data_AA = train_AA, data_Dihedral = sequences, data_lengths = lengths, priors_mu_nu = priors_mu_nu, args = args, include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequence_lengths'])))
    
    test_dic = dic['test']
    sequences = test_dic['sequences_Dihedral']
    lengths = test_dic['sequence_lengths']
    
    
    num_observations = float(lengths.sum())

    
    test_loss = elbo.loss(model, guide, data_DSSP = test_DSSP, data_AA = test_AA, data_Dihedral = test_Dihedral, data_lengths = test_mask, priors_mu_nu = priors_mu_nu, args=args, include_prior=False)
    logging.info('test loss = {}'.format(test_loss / num_observations))
    
    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))
    
    lengths = train_dic['sequence_lengths']
    sequences = train_dic['sequences_Dihedral']
    
    
    
    fakeseq_Dihedral = generate_dihedral_sequence_BIVARIATE(probs_mean1 = pyro.param('AutoDelta.probs_mu'), 
                                                probs_mean2 = pyro.param('AutoDelta.probs_nu'), 
                                                probs_x = pyro.param('AutoDelta.probs_x'),
                                                probs_k1 = pyro.param('AutoDelta.probs_kappa1'),
                                                probs_k2 = pyro.param('AutoDelta.probs_kappa2'),
                                                probs_lam = pyro.param('AutoDelta.probs_lam'))
    
            
            
    plot_rama(fakeseq_Dihedral, plot_title = 'Ramachandran plot', 
              subtitle = 'Data: Generated 20.000 samples given trained model \n Distribution: Bivariate Von Mises Sine model with kappa estimation and informative priors', 
              textstr = 'Hidden states: {}, batch size: {}, nr. steps: {}/{}'.format(args.hidden_dim, args.batch_size, step+1, args.num_steps),
              save_title = 'img5_step{}'.format(step+1))
    
    
    # printing the transition and emission matrix
    for key, value in pyro.get_param_store().items():    
        print(key)
        print(value)





if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=5, type=int)
    parser.add_argument("-tfr", "--test_freq", default=10, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-d", "--hidden-dim", default=40, type=int)
    parser.add_argument("-nrc", "--nr-clusters", default=11, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    parser.add_argument('--tmc', action='store_true',
                        help="Use Tensor Monte Carlo instead of exact enumeration "
                             "to estimate the marginal likelihood. You probably don't want to do this, "
                             "except to see that TMC makes Monte Carlo gradient estimation feasible "
                             "even with very large numbers of non-reparametrized variables.")
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    args = parser.parse_args()
    main(args)

