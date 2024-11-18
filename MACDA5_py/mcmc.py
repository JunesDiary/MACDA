from threesection import get_s11eq_three, get_s21eq_three, get_s31eq_three, get_s41eq_three
from fivesection import get_s11eq_five, get_s21eq_five, get_s31eq_five, get_s41eq_five
from sevensection import get_s11eq_seven, get_s21eq_seven, get_s31eq_seven, get_s41eq_seven
import numpy as np
import math

def model_get(x, freq, s, z_0):    
    returnlist = []
    
    if len(x) == 4:
        for f_ in freq:      
            if s == 's11':   
                ans = float(np.abs(get_s11eq_three(*np.append([f_, z_0, 50], x))))   
            elif s == 's21':  
                ans = float(np.abs(get_s21eq_three(*np.append([f_, z_0, 50], x))))   
            elif s == 's31':   
                ans = float(np.abs(get_s31eq_three(*np.append([f_, z_0, 50], x))))   
            elif s == 's41':     
                ans = float(np.abs(get_s41eq_three(*np.append([f_, z_0, 50], x))))   
            elif s == 's21-s31': 
                s21_temp = float(20 * np.log10(np.abs(get_s21eq_three(*np.append([f_, 50], x)))))
                s31_temp = float(20 * np.log10(np.abs(get_s31eq_three(*np.append([f_, 50], x)))))

                ans = np.abs(s21_temp - s31_temp)

            else:
                ans = 0     
            returnlist.append(ans)   
    
    elif len(x) == 6:
        for f_ in freq:      
            if s == 's11':   
                ans = float(np.abs(get_s11eq_five(*np.append([f_, z_0, 50], x))))   
            elif s == 's21':  
                ans = float(np.abs(get_s21eq_five(*np.append([f_, z_0, 50], x))))   
            elif s == 's31':   
                ans = float(np.abs(get_s31eq_five(*np.append([f_, z_0, 50], x))))   
            elif s == 's41':     
                ans = float(np.abs(get_s41eq_five(*np.append([f_, z_0, 50], x))))   
            elif s == 's21-s31': 
                s21_temp = float(20 * np.log10(np.abs(get_s21eq_five(*np.append([f_, 50], x)))))
                s31_temp = float(20 * np.log10(np.abs(get_s31eq_five(*np.append([f_, 50], x)))))

                ans = np.abs(s21_temp - s31_temp)
            else:
                ans = 0                     
            returnlist.append(ans)  
            
    elif len(x) == 8:
        for f_ in freq:      
            if s == 's11':   
                ans = float(np.abs(get_s11eq_seven(*np.append([f_, z_0, 50], x))))   
            elif s == 's21':  
                ans = float(np.abs(get_s21eq_seven(*np.append([f_, z_0, 50], x))))   
            elif s == 's31':   
                ans = float(np.abs(get_s31eq_seven(*np.append([f_, z_0, 50], x))))   
            elif s == 's41':     
                ans = float(np.abs(get_s41eq_seven(*np.append([f_, z_0, 50], x))))   
            elif s == 's21-s31': 
                s21_temp = float(20 * np.log10(np.abs(get_s21eq_seven(*np.append([f_, 50], x)))))
                s31_temp = float(20 * np.log10(np.abs(get_s31eq_seven(*np.append([f_, 50], x)))))

                ans = np.abs(s21_temp - s31_temp)
            else:
                ans = 0                     
            returnlist.append(ans)  
            
    return (np.array(returnlist))

    
def lnlike(x, freq_solve, s_list, y_, yerr_, z_0):
    ln_alpha = []
    
    for i in range(len(s_list)):
        
        y_i = np.array([y_[i]]*len(freq_solve))
        yerr_i = np.array([yerr_[i]]*len(freq_solve))
               
        model = model_get(x, freq_solve, s_list[i], z_0)       
        
        inv_sigma2 = 1/(yerr_i**2)
        ln_alpha = np.append(ln_alpha, -1*(np.sum((y_i-model)**2*(inv_sigma2/2) - np.log(inv_sigma2/math.sqrt(np.pi*2)))))
    return np.sum(ln_alpha)


def lnprior(x, f_, bounds_, mu_, sigma_, z_0):
    # List to store the computed qi values
    qi = []
    
    # Calculate qi for each frequency
    if len(bounds_) == 4:    
        for freq in f_:
            qi.append(
                float(np.abs((get_s11eq_three(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s21eq_three(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s31eq_three(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s41eq_three(*np.append([freq, z_0, 50], x )))**2)) 
            )
    
    elif len(bounds_) == 6:    
        for freq in f_:
            qi.append(
                float(np.abs((get_s11eq_five(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s21eq_five(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s31eq_five(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s41eq_five(*np.append([freq, z_0, 50], x )))**2)) 
            )
    elif len(bounds_) == 8:    
        for freq in f_:
            qi.append(
                float(np.abs((get_s11eq_seven(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s21eq_seven(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s31eq_seven(*np.append([freq, z_0, 50], x )))**2))  +
                float(np.abs((get_s41eq_seven(*np.append([freq, z_0, 50], x )))**2)) 
            )
            
    # Check if parameters are within bounds (still the same as before)
    conditions_met = 1
    conditions_met &= all((bounds_[i][0] < x[i] < bounds_[i][1]) for i in range(len(x)))
    
    # Check if qi values are close to 1 (still the same as before)
    conditions_met &= all(np.abs(q - 1) < 0.001 for q in qi)
    
    # If any condition is violated, return -inf
    if not conditions_met:
        return -np.inf

    # Applying the Gaussian prior for each parameter x[i]
    gaussian_prior = 0.0
    for i in range(len(x)):
        # For each parameter, apply the Gaussian prior
        if bounds_[i][0] < x[i] < bounds_[i][1]:  # Ensure within bounds
            # Gaussian prior: P(x_i) = exp(-0.5 * ((x_i - mu_i) / sigma_i)^2)
            gaussian_prior += -0.5 * ((x[i] - mu_[i]) / sigma_[i])**2 - 0.5 * np.log(2 * np.pi * sigma_[i]**2)
        else:
            return -np.inf  # Reject if outside bounds
        
    return gaussian_prior


def lnpost(x, freq, s_list_solve, y_, yerr_, bounds_, z_0):    
    
    if len(bounds_) == 4:
        mean = [150,150,150,150]
        sd = [150,150,150,150]
        
    elif len(bounds_) == 6:
        mean = [250,250,250,250,250,250]
        sd = [150,150,150,150,150,150]
    
    elif len(bounds_) == 8:
        mean = [400,400,400,400,400,400,400,400]
        sd = [200,200,200,200,200,200,200,200]
        
    lnp = lnprior(x, freq, bounds_, mean, sd, z_0)    
    if not np.isfinite(lnp):
        return -np.inf    
    templike = lnlike(x, freq, s_list_solve, y_, yerr_, z_0)    
    temp = lnp + templike   
    return temp