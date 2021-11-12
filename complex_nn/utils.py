import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns


def small_training_summary(history):

    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    
    sns.lineplot(x=np.arange(len(history['train_loss'])), y=history['train_loss'], label='training_data', ax=ax[0])
    sns.lineplot(x=np.arange(len(history['val_loss'])), y=history['val_loss'], label='test', ax=ax[0])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Linear loss')
    
    sns.lineplot(x=np.arange(len(history['train_loss'])), y=history['train_loss'], label='training_data', ax=ax[1])
    sns.lineplot(x=np.arange(len(history['val_loss'])), y=history['val_loss'], label='test', ax=ax[1])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    ax[1].set_title('log loss')
    
    sns.lineplot(x=np.arange(len(history['train_acc'])), y=history['train_acc'], label='training_data', ax=ax[2])
    sns.lineplot(x=np.arange(len(history['val_acc'])), y=history['val_acc'], label='test_data', ax=ax[2])
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].set_title('Model Accuracy')

    b


def get_circularity_coeff(dataset, verbose=False):
    
    cov_mat = 0.
    
    # Compute the covariance matrix among real and imaginary parts
    # of the whole dataset
    
    for i in tqdm(range(len(dataset)), leave=False):
        cov_mat += np.cov(dataset[i][0].real.flatten(), dataset[i][0].imag.flatten()) 
        
    # The covariance matrix will have the following structure:  [[Sx**2, Sxy], [Sxy, Sy**2]]
    cov_mat = cov_mat / len(dataset)

    # Compute the variance of Z:   Sz**2 = E[ |Z - E[Z]|**2 ] = Sx**2 + Sy**2
    Sx2 = cov_mat[0,0]
    Sy2 = cov_mat[1,1]
    Sz2 = cov_mat.trace() 
    
    # Get the covariance:   Sxy = E[(X-E[X])(Y-E[Y])]
    Sxy = cov_mat[0,1]
    
    # Compute the pseudo-variance:   Tz = E[ (Z-E[Z])**2 ] = Sx**2 - Sy**2 + 2iSxy
    Tz = Sx2 - Sy2 + 2.j*Sxy
    
    # Compute the circularity quotient:  rhoZ = Tz / Sz**2
    rhoZ = Tz / Sz2
    
    # Compute the correlation coefficient:  rho = Sxy / SxSy
    rho = Sxy / (np.sqrt(Sx2)*np.sqrt(Sy2))
    
    if verbose:
        return cov_mat, Sz2, Tz, rhoZ, rho
    else:
        return rhoZ, rho




