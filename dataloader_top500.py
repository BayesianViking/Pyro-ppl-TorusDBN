# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:30:37 2020

@author: chris
"""

#!/usr/bin/env python3

import numpy as np
from operator import itemgetter 
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader , Subset


def load_500(filename):
    # Amino Acids labels list
    AA_list = []
    # DSSP symbol labels list
    DSSP_list = []
    # Angles list
    list_of_angles = []
    
    for line in filename.readlines():
        if "#" in line:
            continue
        elif "NT" in line:
            continue
        elif "CT" in line:
            continue
        else:
            elemennts_of_line = line.split()
            
            AA_list.append(elemennts_of_line[0])                 # append AA   label
            DSSP_list.append(elemennts_of_line[1])               # append DSSP label
            list_of_angles.append(float(elemennts_of_line[2]))   # append phi
            list_of_angles.append(float(elemennts_of_line[3]))   # append psi
    # list --> numpy       
    angle_array = np.array(list_of_angles)       
    # reshape to numpy:[ phi, psi ]
    angle_array = np.reshape(angle_array,((int(len(angle_array)/2)), 2)) 
        
    return(angle_array,AA_list,DSSP_list)
###############################################################################
# =============================================================================
# # Get Everything from the dataset (apart NT CT regions...)
# all_angles, AA, DSSP = load_500(open(r'C:\Users\Lenovo\Desktop\TH\top500.txt')) # r'C:\Users\Lenovo\Desktop\TH\top500.txt'
#                                                                                  # /home/sgb851/
# =============================================================================

 
################## SENSOR secondary structure #################################
def sensor_or_not(list_of_DSSP, Num_of_classes):
    from collections import Counter
    '''Sensoring and plotting'''
    if  Num_of_classes == 3:       
        '''Sensoring METHOD A'''
        # B to E
        list_of_DSSP = [dssp.replace('B', 'E') for dssp in list_of_DSSP]
        # G to E
        list_of_DSSP = [dssp.replace('G', 'H') for dssp in list_of_DSSP]
        
        # Rest to coil
        list_of_DSSP = [dssp.replace('S', 'C') for dssp in list_of_DSSP]
        list_of_DSSP = [dssp.replace('T', 'C') for dssp in list_of_DSSP]
        list_of_DSSP = [dssp.replace('-', 'C') for dssp in list_of_DSSP]
        list_of_DSSP = [dssp.replace('I', 'C') for dssp in list_of_DSSP]
        
        keys_DSSP = Counter(list_of_DSSP).keys()      
        freqs_DSSP =  Counter(list_of_DSSP).values() 
        
        # Turn to np.array
        list_of_DSSP = np.array(list_of_DSSP)
        list_of_DSSP[list_of_DSSP=='E'] = 0           # E=0
        list_of_DSSP[list_of_DSSP=='H'] = 1           # H=1
        list_of_DSSP[list_of_DSSP=='C'] = 2           # C=2
          
        list_of_DSSP = list_of_DSSP.astype(int)
        ###############################################################################
        y_pos=list(range(0,Num_of_classes))
        # Create bars
        plt.bar(y_pos, freqs_DSSP , width = 0.5, color = 'b')
        # Create names on the x-axis
        plt.xticks(y_pos,  keys_DSSP)
        plt.title("Frequencies of DSSP symbols in dataset")
        #restart Kernel in Error: TypError:'str' object is not callable'
        # Show graphic
        fig = plt.figure()
        ###############################################################################
        
    elif Num_of_classes == 8:
        '''Keep the 8 classes'''
        keys = Counter(list_of_DSSP).keys()      
        freqs =  Counter(list_of_DSSP).values()
        # Turn to np.array
        list_of_DSSP = np.array(list_of_DSSP)
        list_of_DSSP[list_of_DSSP=='B'] = 0
        list_of_DSSP[list_of_DSSP=='E'] = 1
        list_of_DSSP[list_of_DSSP=='G'] = 2
        list_of_DSSP[list_of_DSSP=='H'] = 3
        list_of_DSSP[list_of_DSSP=='S'] = 4
        list_of_DSSP[list_of_DSSP=='T'] = 5
        list_of_DSSP[list_of_DSSP=='-'] = 6
        list_of_DSSP[list_of_DSSP=='I'] = 7
        
        list_of_DSSP = list_of_DSSP.astype(int)
        ###############################################################################
        y_pos=list(range(0,Num_of_classes))
        # Create bars
        plt.bar(y_pos, freqs, width = 0.5, color = 'b')
        # Create names on the x-axis
        plt.xticks(y_pos,  keys)
        plt.title("Frequencies of DSSP symbols in dataset")
        # Show graphic
        fig = plt.figure()
        print(" 'I' has only 20 occurencies in whole dataset")
        ###############################################################################
        

    else:
        print('Error: 3 or 8 classes of labels')
        
    return(list_of_DSSP, fig)


# =============================================================================
# DSSP,fig = sensor_or_not(DSSP,3)
# =============================================================================

# METHOD B ...
# METHOD C ....

################# AA to numbers ##################################
def AA_to_num(list_of_AA):
    '''A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V'''
    # Check for imbalance in data labels                         
    # counts the elements' frequency
#    print(Counter(list_of_AA).keys())
#    print("AA freqs in dataset:", Counter(list_of_AA).values())
    keys_AA = Counter(list_of_AA).keys()      
    freqs_AA =  Counter(list_of_AA).values() 
#    print("Check correspondance of freq with spesific AA", list_of_AA.count('C') )    
    
    for i in range(0, len(list_of_AA)):
        if list_of_AA[i].startswith('A'):
            list_of_AA[i] = int(0)
        elif list_of_AA[i].startswith('R'):
            list_of_AA[i] = int(1)
        elif list_of_AA[i].startswith('N'):
            list_of_AA[i] = int(2)
        elif list_of_AA[i].startswith('D'):
            list_of_AA[i] = int(3)
        elif list_of_AA[i].startswith('C'):
            list_of_AA[i] = int(4)
        elif list_of_AA[i].startswith('Q'):
            list_of_AA[i] = int(5)       
        elif list_of_AA[i].startswith('E'):
            list_of_AA[i] = int(6)
        elif list_of_AA[i].startswith('G'):
            list_of_AA[i] = int(7)
        elif list_of_AA[i].startswith('H'):
            list_of_AA[i] = int(8)       
        elif list_of_AA[i].startswith('I'):
            list_of_AA[i] = int(9)
        elif list_of_AA[i].startswith('L'):
            list_of_AA[i] = int(10)
        elif list_of_AA[i].startswith('K'):
            list_of_AA[i] = int(11)       
        elif list_of_AA[i].startswith('M'):
            list_of_AA[i] = int(12)
        elif list_of_AA[i].startswith('F'):
            list_of_AA[i] = int(13)
        elif list_of_AA[i].startswith('P'):
            list_of_AA[i] = int(14)
        elif list_of_AA[i].startswith('S'):
            list_of_AA[i] = int(15)
        elif list_of_AA[i].startswith('T'):
            list_of_AA[i] = int(16)
        elif list_of_AA[i].startswith('W'):
            list_of_AA[i] = int(17)
        elif list_of_AA[i].startswith('Y'):
            list_of_AA[i] = int(18)
        elif list_of_AA[i].startswith('V'):
            list_of_AA[i] = int(19)        

    list_of_AA = np.array(list_of_AA)
    list_of_AA = list_of_AA.astype(int)
    y_pos=list(range(0,20))
    # Create bars
    plt.bar(y_pos, freqs_AA, color = 'purple')
    # Create names on the x-axis
    plt.xticks(y_pos, keys_AA)
    plt.title("Frequencies of AAs in dataset")
    # Show graphic
    fig=plt.figure()   
    return(list_of_AA,fig)

# =============================================================================
# AA, fig = AA_to_num(AA)
# =============================================================================

##############################################################################

# Create dataset class, so we can use Pytorch's Dataloader functionalities
class MyDataset( Dataset ):
    def __init__(self, Angles, AA,DSSP, transform=None):
        # torch.tensor() always copies data.
        # If you have a NumPy ndarray and want to avoid a copy, use torch.from_numpy().      
        self.Angles = torch.from_numpy(Angles).float()
        
        
#        self.AA = torch.from_numpy(AA).int()
        self.AA = torch.from_numpy(AA).long()
        # create one hot labels
        self.AA=torch.zeros(len(self.AA), (self.AA).max()+1).scatter_(1, (self.AA).unsqueeze(1), 1.)
        
        
#        self.DSSP = torch.from_numpy(DSSP).int()
        self.DSSP = torch.from_numpy(DSSP).long()
        # create one hot labels
        self.DSSP=torch.zeros(len(self.DSSP), (self.DSSP).max()+1).scatter_(1, (self.DSSP).unsqueeze(1), 1.)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.Angles[index]
        y1 = self.AA[index]
        y2 = self.DSSP[index]
        if self.transform:
            x = self.transform(x)
        
        return x, y1, y2
    
    def __len__(self):
        return len(self.Angles)
    
    
# =============================================================================
# DATASET = MyDataset( all_angles, AA, DSSP )     
# =============================================================================


                                                                                #____________36936936936936936
                                                                                #____________36936936936936936
                                                                                #____________369369369369369369
                                                                                #___________36936936936936933693
                                                                                #__________3693693693693693693693
                                                                                #_________369369369369369369369369
                                                                                #_________3693693693693693693693699
                                                                                #________3693693693693693693693699369
                                                                                #_______36936939693693693693693693693693
                                                                                #_____3693693693693693693693693693693636936
                                                                                #___36936936936936936936936936936___369369369
                                                                                #__36936___369336936369369369369________36936
                                                                                #_36936___36936_369369336936936___
                                                                                #36933___36936__36936___3693636__
                                                                                #693____36936__36936_____369363_
                                                                                #______36936__36936______369369__
                                                                                #_____36936___36936_______36936___
                                                                                #_____36936___36936________36936___
                                                                                #_____36936___36936_________36936___
                                                                                #______369____36936__________369__
                                                                                #______________369________________
                                                                                #_______________________________

                                                                               # CHECK again for CUDA      work, pin
# Setup Pytorch dataloaders                                         
def setup_data_loaders( dataset, batchsize ):
    '''Separate 80/20 the dataset  
    Return: train/TEST loader , test set'''                                                    # CHECK again how to properly split the dataset-- StratifiedShuffleSplit  ??
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    print (" ------------------------------------------------ ")
    print (" Size of train set:",  len(train_set)) 
    print (" Size of TEST set:",   len(test_set))   
    print (" Size of each batch:", batchsize,"examples")                                                                                   
    train_loader = DataLoader( train_set, 
                        batch_size=batchsize, 
                        shuffle=True,                                               
                        num_workers=0, 
                        pin_memory=False )  # Running on CPU
    
    TEST_loader = DataLoader( test_set, 
                        batch_size=batchsize, 
                        shuffle=True,                                               
                        num_workers=0, 
                        pin_memory=False)  # Running on CPU
    return(train_loader,TEST_loader, test_set )


# =============================================================================
# # GET train/test data in dataloader form
# train_loader, TEST_loader, test_set = setup_data_loaders( DATASET, 200)
# 
# =============================================================================
# =============================================================================
# # Check size of train/test batch
# print ('train loader size:', len(train_loader))
# print ('TEST loader size:',  len(TEST_loader))
# 
# ################## Check sizes of data- tensors #################################
# #for batch_idx, ( all_angles, AA, DSSP ) in enumerate(train_loader):  # arg: train_loader or TEST_loader
# #    print('Batch idx {}, Angles {}, AA {}, DSSP {}'.format(
# #        batch_idx, all_angles.shape, AA.shape, DSSP.shape))
# ##############################################################################
# # print shapes of dataset tensors
# print (" ------------------------------------------------ ")
# print ("  --> Size of whole dataset:", len(DATASET))
# print ("      Angles:", DATASET.Angles.shape)
# print ("      AA: ",    DATASET.AA.shape)
# print ("      DSSP:",   DATASET.DSSP.shape)
# print (" ------------------------------------------------ ")
# =============================================================================
    
def Rama1( dataset ):
    '''makes Ramachandran plot '''
    from matplotlib.colors import LogNorm
    #convert radians to degrees
    Degrees = np.rad2deg(dataset) 
    # Get phi, psi angles from dataset
    phi = Degrees[:,0] 
    psi = Degrees[:,1] 
    plt.figure(figsize=(6, 6))
    plt.hist2d( phi, psi, bins = 200, norm = LogNorm(), cmap = plt.cm.jet )
    plt.title('Rama')

    plt.xlabel('phi')
    plt.ylabel('psi')
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    fig = plt.figure()
    return(fig)    
    
def Rama2( dataset ):
    '''makes Ramachandran plot '''
    from scipy.stats import kde
    #convert radians to degrees
    Degrees = np.rad2deg(dataset) 
    # Get phi, psi angles from dataset
    phi = Degrees[:,0] 
    psi = Degrees[:,1] 
    
    nbins=100
    k = kde.gaussian_kde([phi,psi])
    xi, yi = np.mgrid[phi.min():phi.max():nbins*1j, psi.min():psi.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    # Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
 
    plt.title('Rama')
    plt.xlabel('phi')
    plt.ylabel('psi')


    fig = plt.figure()
    return(fig)
   

def Specific_Amino_Rama( dataset, AminoAcids, title ):
    '''makes Ramachandran plots based on the 
    specific amino acids-angles taken from dataset:
    A , R , N , D , C , Q , E , G , H , I , L  , K  , M  , F  , P  , S  , T  , W  , Y  , V
    0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8, 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19
    input:  -dataset created from MyDataset Class
            -list of spesific amino acids
            -title of graph
        '''
    amino_acid_list = []
    # iterate over amino acids in input: list of spesific amino acids
    for AA in AminoAcids:
        # Get index of spesific amino acid in dataset 
        for i,x in enumerate(dataset.AA):
            if x==AA:
                amino_acid_list.append(i)
               
    # Get phi,psi angles based on the index list above
    angles = list(itemgetter(*amino_acid_list)(dataset.Angles.numpy()))
    # Convert to numpy
    angles = np.array(angles)
    # Convert radians to degrees
    degrees = np.rad2deg(angles) 
    
    # Get phi, psi angles from dataset
    phi = degrees[:,0] 
    psi = degrees[:,1] 
    plt.scatter(phi, psi, s=1)
    plt.title( title)
    plt.xlabel('phi')
    plt.ylabel('psi')
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    fig=plt.figure()
    return(fig)
    

# =============================================================================
# # Plotting time
# plt.show(Whole_Rama( DATASET.Angles ))    
# plt.show(Specific_Amino_Rama( DATASET, [ 14 ] , title = 'P' ))
# plt.show(Specific_Amino_Rama( DATASET, [ 7  ],  title = 'G' ))
# =============================================================================

############################################################################################################################################################

def plot_ELBOs(train_elbo, test_elbo,test_frequency):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    plt.figure(figsize=(8, 8))
#    sns.set_style("white")
#    sns.set_style("ticks")
#    sns.despine()
    sns.set_style("whitegrid")
    test_elbo=np.array(test_elbo)
    train_elbo=np.array(train_elbo)
    list_of_epochs=[]
    for i in range(0,len(train_elbo),test_frequency):
        list_of_epochs.append(i)
    list_of_epochs = np.array(list_of_epochs)
    # for test ELBO
    data = np.concatenate([list_of_epochs[:, sp.newaxis], test_elbo[:, sp.newaxis]], axis=1)
    # for training ELBO
    data2 = np.concatenate([np.arange(len(train_elbo))[:, sp.newaxis], train_elbo[:, sp.newaxis]], axis=1)
    # Plot every N training epochs the test ELBO->Test Frequency
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Test ELBO'])
    df2 = pd.DataFrame(data=data2, columns=['Training Epoch', 'Train ELBO'])
    plt.plot( 'Training Epoch', 'Test ELBO', data=df, marker='o', markerfacecolor='blue', markersize=2, color='skyblue', linewidth=4)
    plt.plot( 'Training Epoch', 'Train ELBO', data=df2, marker='o', markerfacecolor='red', markersize=2, color='red', linewidth=4)
    plt.title('Training/Testing ELBO',fontsize=14,loc='center')
    plt.show()
    
#    g = sns.FacetGrid(df, height=10, aspect=1.5)
#    g = sns.FacetGrid(df2, height=10, aspect=1.5)
#    g.map(plt.scatter, "Training Epoch", "Test ELBO")
#    g.map(plt.scatter, "Training Epoch", "Train ELBO")
#    g.map(plt.plot, "Training Epoch", "Test ELBO")
#    g.map(plt.plot, "Training Epoch", "Train ELBO")
#    plt.show('test_elbo_vae.png') #./
    sns.set_style("ticks")

#    plt.savefig('./vae_results/test_elbo_vae.png') #./
#    plt.close('all')










