import sympy as sp
import math
import itertools
from sympy.matrices import Matrix
from sympy.abc import f
import pickle


def create_symbols(num_sections):
    symbols_list_v = {}
    symbols_list_h = {}
    
    for i in range(1, ((num_sections + 2) // 2) + 1):
        symbols_list_v[f'zv_{i}'] = sp.symbols(f'zv_{i}')
        
    for i in range(1, ((num_sections + 1) // 2) + 1):
        symbols_list_h[f'zh_{i}'] = sp.symbols(f'zh_{i}')
        
    # Combine the values from both dictionaries into a single list
    symbols_list = [x for x in itertools.chain(*itertools.zip_longest(symbols_list_v.values(), symbols_list_h.values())) if x is not None]

    return symbols_list

def unitcell(z_alpha, z_beta, z_gamma, chooseodd, invert, eps_r, f_res): #if invert is True - the unit cell is flipped about vertical line of synmetry   
    bl_uc = (math.pi * sp.sqrt(eps_r) * f) / (8 * f_res)
    
    #alpha component of unit cell, always a TL
    abcd_tl_alpha = Matrix([[sp.cos(bl_uc), (sp.I) * z_alpha  * sp.sin(bl_uc)],
                      [((sp.I) * sp.sin(bl_uc))/z_alpha, sp.cos(bl_uc)]])  
    #beta component of unit cell, which is always stub, hence can be even or odd
    if chooseodd:
        abcd_stub_beta = Matrix([[1, 0],
                                 [(-sp.I)/(z_beta*sp.tan(bl_uc)), 1]])
    else:
        abcd_stub_beta = Matrix([[1, 0],
                                 [(sp.I * sp.tan(bl_uc))/z_beta, 1]])
    
    #gamma component of unit cell, always a TL
    abcd_tl_gamma = Matrix([[sp.cos(bl_uc), (sp.I) * z_gamma  * sp.sin(bl_uc)],
                  [((sp.I) * sp.sin(bl_uc))/z_gamma, sp.cos(bl_uc)]]) 
    
    #helpful argument while joining two unit cells into a square
    if invert:
        abcd = (abcd_tl_gamma * abcd_stub_beta) * abcd_tl_alpha
    else:
        abcd = (abcd_tl_alpha * abcd_stub_beta) * abcd_tl_gamma
    #print(simplify(abcd))
    #print('\n\n')
    return (abcd)


def create_abcd_eq(symbols_list, z0, eps_r, f_res):
    # chooseodd  invert
    imp_list = [z0] + symbols_list + symbols_list[::-1][1:] + [z0] 

    for i in range(0,len(imp_list)-1,2):
        
        #print(imp_list[i], imp_list[i+1], imp_list[i+2])
        if i == 0:
            abcd_even = unitcell(imp_list[i], imp_list[i+1], imp_list[i+2], False, False, eps_r, f_res)
            abcd_odd = unitcell(imp_list[i], imp_list[i+1], imp_list[i+2], True, False, eps_r, f_res)
        else:
            abcd_even = abcd_even * unitcell(imp_list[i], imp_list[i+1], imp_list[i+2], False, False, eps_r, f_res)
            abcd_odd = abcd_odd * unitcell(imp_list[i], imp_list[i+1], imp_list[i+2], True, False, eps_r, f_res)



    a_e = abcd_even.row(0)[0]
    b_e = abcd_even.row(0)[1]
    c_e = abcd_even.row(1)[0]
    d_e = abcd_even.row(1)[1]

    a_o = abcd_odd.row(0)[0]
    b_o = abcd_odd.row(0)[1]
    c_o = abcd_odd.row(1)[0]
    d_o = abcd_odd.row(1)[1]

    a_e, b_e = abcd_even.row(0)[0], abcd_even.row(0)[1]
    c_e, d_e = abcd_even.row(1)[0], abcd_even.row(1)[1]
    
    a_o, b_o = abcd_odd.row(0)[0], abcd_odd.row(0)[1]
    c_o, d_o = abcd_odd.row(1)[0], abcd_odd.row(1)[1]
    
    # Store variables in a dictionary
    abcd_vars = {
        'a_e': a_e, 'b_e': b_e, 'c_e': c_e, 'd_e': d_e,
        'a_o': a_o, 'b_o': b_o, 'c_o': c_o, 'd_o': d_o
    }
    
    # Save the dictionary to a file
    with open('abcd_vars.pkl', 'wb') as f:
        pickle.dump(abcd_vars, f)

    print(f"Sympy equations temporarily saved to abcd_vars.pkl\n")

    return 