import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fox_toolbox.utils import xml_parser

        
def calib_plot(sigma, debug, irsmout=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30,8))

    expiries = sigma.buckets[:-1]
    sigmas = sigma.values[:-1]

    ax1.step(expiries,   sigmas,   label = 'nk.Henrard', marker='o', where='pre');
    
    if irsmout:
        hw_params = xml_parser.get_hw_params(irsmout)
        ref_sigmas = hw_params.hw_volatility.values[:-1]
        ax1.step(expiries, ref_sigmas, label = 'Macs', linestyle='--', where='pre');
        ax11 = ax1.twinx()
        ax11.plot(expiries, ref_sigmas -sigmas, ls=':')
        ax11.set_ylabel('difference', fontsize=16);
    
    ax1.set_ylim(0);
    ax1.legend(loc='lower right');
    ax1.set_title('calibrated Hull White piecewise constant volatility', fontsize=20);
    ax1.set_xlabel('expiries', fontsize=16);
    ax1.set_ylabel('volatility', fontsize=16);


    model_prices = np.array(debug['model_price'])
    target_prices = np.array(debug['target_price'])
    
    ax2.plot(expiries[1:], model_prices,  label = 'model_price', marker = 'o');
    ax2.plot(expiries[1:], target_prices,  label = 'target_price', linestyle = '--');
    

    ax2.set_title('Swaption prices', fontsize=20);
    ax2.legend(loc='upper right');
    ax2.set_xlabel('expiries', fontsize=16);
    ax2.set_ylabel('Prices per 1$', fontsize=16);

    ax22 = ax2.twinx()
    ax22.plot(expiries[1:], model_prices - target_prices, linestyle=':' )
    ax22.set_ylabel('difference', fontsize=16)
    
    

