def print_metadata():
    metadata = """
    ================================
    Script: MCMC Asisted Coupler Designer Analyzer
    Version: 5.0
    Authors: Arjun Ghosh, Ritoban Basu Thakur
    Last Update: 2024-11-07
    
    Solve for 3,5,7 section butterworth quadrature
    3dB Couplers which are symmetric
    
    Vers5.0--
    *Manual fine tuning after results from run
    *Gaussian Priors applied.
    ================================
    """
    
    macda_ascii = """
     M     M    AAAAA     CCCCC     DDDDD     AAAAA  
     MM   MM   A     A   C         D     D   A     A 
     M M M M   AAAAAAA   C         D     D   AAAAAAA 
     M  M  M   A     A   C         D     D   A     A 
     M     M   A     A   CCCCC     DDDDD     A     A
    """
    
    print(metadata)
    print(macda_ascii)

# Call the print_metadata function at the start of the script
print_metadata()


from imports import *
from createblocks import create_abcd_eq, create_symbols
from threesection import numba_precompile_three, qhdc_plotter_three, get_s11eq_three, get_s21eq_three, get_s31eq_three, get_s41eq_three
from fivesection import numba_precompile_five, qhdc_plotter_five, get_s11eq_five, get_s21eq_five, get_s31eq_five, get_s41eq_five
from sevensection import numba_precompile_seven, qhdc_plotter_seven, get_s11eq_seven, get_s21eq_seven, get_s31eq_seven, get_s41eq_seven
from mcmc import lnpost
from analysis import impedance_tweaker, save_parameter_timeseries, save_corner_plot, save_emcee_summary




#==================================================
s11_actual = 1e-06   #  s41_actual is the same
s21_actual = 0 - 0.501j
s31_actual = -0.501 + 0.j

s11s41_ab = np.abs(s11_actual)
s11s41_ph = np.angle(s11_actual)

s21_ab = np.abs(s21_actual)
s21_ph = np.angle(s21_actual)

s31_ab = np.abs(s31_actual)
s31_ph = np.angle(s31_actual)

s21_s31_db = 0.1
#==================================================

#input
eps_r = int(input("Enter the eps_r (default is 4.0): ") or 3)
f_res = int(input("Enter the f_res (default is 25e9 Hz): ") or 25e9)

z_0 = int(input("Enter the z_0 (default is 50 Ohm): ") or 50)

z_line = z_0/math.sqrt(2)

f, z0 = sp.symbols('f z0')

# Input: number of sections from the user
num_sections = int(input("Enter the number of sections (default is 3): ") or 3)

# Generate the symbols
global symbols_list
symbols_list = create_symbols(num_sections)
print('Variables solving for: '+str(symbols_list))


_ = create_abcd_eq(symbols_list=symbols_list, z0=z0, eps_r=eps_r, f_res=f_res)

# Load the variables from the pickle file as strings
with open('abcd_vars.pkl', 'rb') as f:
    abcd_vars = pickle.load(f)

# List of variable names
equations = ['a_e', 'a_o', 'b_e', 'b_o', 'c_e', 'c_o', 'd_e', 'd_o']

# Loop through each equation and process it
for eq in equations:
    start = time.time()
    length = len(str(abcd_vars[eq]))
    eq_numba = str(abcd_vars[eq]).replace('cos(', 'np.cos(').replace('sin(', 'np.sin(').replace('tan(', 'np.tan(')
    end = time.time()
    print(f'{eq} equation ({length} char) loaded in {round(end - start, 4)} s')

os.remove('abcd_vars.pkl')
print('\nabcd_vars.pkl file deleted')

import sys
#run numba precompile for a three section in this case

if num_sections == 3:
    numba_precompile_three()
    
elif num_sections == 5:
    numba_precompile_five()

elif num_sections == 7:
    numba_precompile_seven()

def get_input(prompt, default_value, value_type=float):
    # Clear the input buffer to ensure a clean input prompt
    sys.stdout.flush()
    time.sleep(0.3)  # Wait for the terminal to process
    user_input = input(f"{prompt} (default {default_value}): ")
    if user_input.lower() == "all":
        return "all"
    
    if not user_input:
        return default_value
    
    return value_type(user_input)

# Function to set default values for all fields fo 3 section
def set_default_values_3():
    return (
        [250, 100, 100, 100],
        [[0, 300], [0, 300], [0, 300], [0, 300]],
        np.array([[0.1, 0.1, 0.1] for _ in range(4)]),
        [20e9, 25e9, 30e9],
        4,
        ['s11', 's21', 's31', 's41'],
        30,
        10000,
        0.001,
        7000,
        5,
        30
    )

# Function to set default values for all fields
def set_default_values_5():
    return (
        [350,350,350,350,350,350],
        [[0, 451], [0, 451], [0, 451], [0, 451], [0,451], [0,451]],
        np.array([[0.1, 0.1, 0.1, 0.1, 0.1] for _ in range(4)]),
        [15e9, 20e9, 25e9, 30e9, 35e9],
        6,
        ['s11', 's21', 's31', 's41'],
        350,
        60000,
        0.001,
        9000,
        5,
        30
    )

# Function to set default values for all fields
def set_default_values_7():
    return (
        [350,350,350,350,350,350,350,350],
        [[0,700],[0,700],[0,700],[0,700],[0,700],[0,700],[0,700],[0,700]],
        np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] for _ in range(4)]),
        [12e9, 15e9, 20e9, 25e9, 30e9, 35e9, 38e9],
        8,
        ['s11', 's21', 's31', 's41'],
        350,
        30000,
        0.001,
        10000,
        5,
        30
    )

# First input field check for "all" (if "all", set default values for everything)
if num_sections == 3:
    first_input = get_input("Enter guess value for parameter 1", 250)
elif num_sections == 5:
    first_input = get_input("Enter guess value for parameter 1", 350)
elif num_sections == 7:
    first_input = get_input("Enter guess value for parameter 1", 400)
    
if first_input == "all":
    if num_sections == 3:
        guess, bounds, y_err_tol_matrix, freq_solve, ndim, s_list_solve, nwalkers, nsteps, pert, burn_in, thin, noofcores =         set_default_values_3()
    elif num_sections == 5:
        guess, bounds, y_err_tol_matrix, freq_solve, ndim, s_list_solve, nwalkers, nsteps, pert, burn_in, thin, noofcores =         set_default_values_5()
    elif num_sections == 7:
        guess, bounds, y_err_tol_matrix, freq_solve, ndim, s_list_solve, nwalkers, nsteps, pert, burn_in, thin, noofcores =         set_default_values_7()
        
        
else:
    if num_sections == 3:
        # Taking inputs from the user
        guess = [
            first_input,
            get_input("Enter guess value for parameter 2", 150),
            get_input("Enter guess value for parameter 3", 150),
            get_input("Enter guess value for parameter 4", 150)
        ]
        
        # Taking bounds as input
        bounds = [
            [get_input("Enter lower bound for parameter 1", 0), get_input("Enter upper bound for parameter 1", 300)],
            [get_input("Enter lower bound for parameter 2", 0), get_input("Enter upper bound for parameter 2", 300)],
            [get_input("Enter lower bound for parameter 3", 0), get_input("Enter upper bound for parameter 3", 300)],
            [get_input("Enter lower bound for parameter 4", 0), get_input("Enter upper bound for parameter 4", 300)],
            
        ]
        
        # Taking the frequency solve points as input
        freq_solve = [
            get_input("Enter frequency point 1", 20e9),
            get_input("Enter frequency point 2", 25e9),
            get_input("Enter frequency point 3", 30e9)
        ]
        
        
    elif num_sections == 5:
        # Taking inputs from the user
        guess = [
            first_input,
            get_input("Enter guess value for parameter 2", 450),
            get_input("Enter guess value for parameter 3", 450),
            get_input("Enter guess value for parameter 4", 450),
            get_input("Enter guess value for parameter 5", 450),
            get_input("Enter guess value for parameter 6", 450)
        ]
        
        # Taking bounds as input
        bounds = [
            [get_input("Enter lower bound for parameter 1", 0), get_input("Enter upper bound for parameter 1", 451)],
            [get_input("Enter lower bound for parameter 2", 0), get_input("Enter upper bound for parameter 2", 451)],
            [get_input("Enter lower bound for parameter 3", 0), get_input("Enter upper bound for parameter 3", 451)],
            [get_input("Enter lower bound for parameter 4", 0), get_input("Enter upper bound for parameter 4", 451)],
            [get_input("Enter lower bound for parameter 5", 0), get_input("Enter upper bound for parameter 5", 451)],
            [get_input("Enter lower bound for parameter 6", 0), get_input("Enter upper bound for parameter 6", 451)]
        ]
        
        # Taking the frequency solve points as input
        freq_solve = [
            get_input("Enter frequency point 1", 15e9),
            get_input("Enter frequency point 2", 20e9),
            get_input("Enter frequency point 3", 25e9),
            get_input("Enter frequency point 4", 30e9),
            get_input("Enter frequency point 5", 35e9)
        ]

    elif num_sections == 7:
        # Taking inputs from the user
        guess = [
            first_input,
            get_input("Enter guess value for parameter 2", 350),
            get_input("Enter guess value for parameter 3", 350),
            get_input("Enter guess value for parameter 4", 350),
            get_input("Enter guess value for parameter 5", 350),
            get_input("Enter guess value for parameter 6", 350),
            get_input("Enter guess value for parameter 7", 350),
            get_input("Enter guess value for parameter 8", 350)
        ]
        
        # Taking bounds as input
        bounds = [
            [get_input("Enter lower bound for parameter 1", 0), get_input("Enter upper bound for parameter 1", 700)],
            [get_input("Enter lower bound for parameter 2", 0), get_input("Enter upper bound for parameter 2", 700)],
            [get_input("Enter lower bound for parameter 3", 0), get_input("Enter upper bound for parameter 3", 700)],
            [get_input("Enter lower bound for parameter 4", 0), get_input("Enter upper bound for parameter 4", 700)],
            [get_input("Enter lower bound for parameter 5", 0), get_input("Enter upper bound for parameter 5", 700)],
            [get_input("Enter lower bound for parameter 6", 0), get_input("Enter upper bound for parameter 6", 700)],
            [get_input("Enter lower bound for parameter 7", 0), get_input("Enter upper bound for parameter 7", 700)],
            [get_input("Enter lower bound for parameter 8", 0), get_input("Enter upper bound for parameter 8", 700)]
        ]
        
        # Taking the frequency solve points as input
        freq_solve = [
            get_input("Enter frequency point 1", 12e9),            
            get_input("Enter frequency point 2", 15e9),
            get_input("Enter frequency point 3", 20e9),
            get_input("Enter frequency point 4", 25e9),
            get_input("Enter frequency point 5", 30e9),
            get_input("Enter frequency point 6", 35e9),
            get_input("Enter frequency point 7", 38e9)
            
        ]

        
ndim = int(get_input("Enter number of parameters", 6, value_type=int))
# Taking the y_err_tol_matrix as input
y_err_tol_matrix = np.array([
    [[get_input(f"Enter value for y_err_tol_matrix[0, {i}]", 10**-1)]*(ndim-1) for i in range(3)],
    [[get_input(f"Enter value for y_err_tol_matrix[1, {i}]", 10**-1)]*(ndim-1) for i in range(3)],
    [[get_input(f"Enter value for y_err_tol_matrix[2, {i}]", 10**-1)]*(ndim-1) for i in range(3)],
    [[get_input(f"Enter value for y_err_tol_matrix[3, {i}]", 10**-1)]*(ndim-1)  for i in range(3)]
])

    

s_list_solve = input("Enter S-parameters to solve (comma-separated, e.g., 's11, s21')").split(",")
nwalkers = int(get_input("Enter number of walkers", 30, value_type=int))
nsteps = int(get_input("Enter number of steps", 50000, value_type=int))
pert = get_input("Enter perturbation value", 0.001)
burn_in = int(get_input("Enter burn-in steps", 9000, value_type=int))
thin = int(get_input("Enter thinning factor", 5, value_type=int))
noofcores = int(get_input("Enter number of cores", 32, value_type=int))

    

y = np.array([
    
[s11s41_ab]*(ndim-1),
[s21_ab]*(ndim-1),
[s31_ab]*(ndim-1),
[s11s41_ab]*(ndim-1)
    
])

# Print out the collected values
print("\nCollected values:")
print(f"Guess: {guess}")
print(f"Bounds: {bounds}")
print(f"y_err_tol_matrix: {y_err_tol_matrix}")
print(f"y: {y}")
print(f"Frequency Solve Points: {freq_solve}")
print(f"Number of parameters (ndim): {ndim}")
print(f"S-parameters to solve: {s_list_solve}")
print(f"Number of walkers: {nwalkers}")
print(f"Number of steps: {nsteps}")
print(f"Perturbation value: {pert}")
print(f"Burn-in steps: {burn_in}")
print(f"Thinning factor: {thin}")
print(f"Number of cores: {noofcores}")

# Generate the date-time and filename string
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"meta_{date_time}_w{nwalkers}_s{nsteps}_{len(freq_solve)}section"

# Define the path for the new folder
if os.path.exists(folder_name):
    # Delete the folder if it already exists
    os.rmdir(folder_name)
os.makedirs(folder_name)

# Define the path for the readme file within the folder
readme_path = os.path.join(folder_name, "readme.txt")

# Saving the collected values to a metadata file
with open(readme_path, "w") as file:
    file.write("Run Metadata\n")
    file.write("================\n")
    file.write(f"Guess: {guess}\n")
    file.write(f"Bounds: {bounds}\n\n")
    file.write(f"y_err_tol_matrix: {y_err_tol_matrix}\n")
    file.write(f"y: {y}\n\n")
    file.write(f"Frequency Solve Points: {freq_solve}\n")
    file.write(f"Number of parameters (ndim): {ndim}\n")
    file.write(f"S-parameters to solve: {s_list_solve}\n")
    file.write(f"Number of walkers: {nwalkers}\n")
    file.write(f"Number of steps: {nsteps}\n")
    file.write(f"Perturbation value: {pert}\n")
    file.write(f"Burn-in steps: {burn_in}\n")
    file.write(f"Thinning factor: {thin}\n")
    file.write(f"Number of cores: {noofcores}\n")

print(f"\nMetadata saved to {readme_path}.")

p0 = [guess + pert * np.random.randn(ndim) for i in range(nwalkers)]

start = time.time()

# Use multiprocessing to run the MCMC
with Pool(processes=4) as pool:  # Use the number of cores you want
    # Call the lnpost function from mcmc.py inside the emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(freq_solve, s_list_solve, y, y_err_tol_matrix, bounds, z_0), pool=pool)
    sampler.run_mcmc(p0, nsteps, progress=True)

end = time.time()
multi_time = end - start
print(f"Multiprocessing took {multi_time:.1f} seconds")



# Assuming 'sampler' is your emcee.EnsembleSampler object
samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)  # Discard the burn-in samples and thin the chain
 # Calculate the mean of the posterior distributions
mean_values = np.mean(samples, axis=0)
print("Mean values:", mean_values)
# Calculate the median of the posterior distributions
median_values = np.median(samples, axis=0)
print("Median values:", median_values)
# Calculate the standard deviation of the posterior distributions
std_devs = np.std(samples, axis=0)
print("Standard deviations:", std_devs)
# If you want to find the MAP estimate (the sample with the highest posterior probability)
# Note: This requires the log_prob_fn to be the posterior probability function
log_probs = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
max_prob_index = np.argmax(log_probs)
map_estimate = samples[max_prob_index]
print("MAP estimate:", map_estimate)


# Saving the result data to a metadata file
with open(readme_path, "a") as file:
    file.write("\n\nRun Results\n")
    file.write("================================\n")
    file.write(f"\nMean values: {mean_values}\n"    )
    file.write(f"Median values: {median_values}\n")
    file.write(f"Standard deviations: {std_devs}\n")
    file.write(f"MAP estimate: {map_estimate}\n")

print(f"\nRun results saved to {readme_path}.")

# if num_sections == 3:
#     _, _ = qhdc_plotter_three(map_estimate, eps_r, f_res, z_0)

try:
    impedance_tweaker(mean_values, f_res)
    print('Tweak with the design running')
except:
    print('Tweak with the design failed, drawing plot with map_estimate and median_values')
    if num_sections == 3:
        _, _ = qhdc_plotter_three(map_estimate, eps_r, f_res, z_0)
        _, _ = qhdc_plotter_three(mean_values, eps_r, f_res, z_0)


samples_whole = sampler.get_chain()
save_parameter_timeseries(samples_whole, folder_name, symbols_list, map_estimate, mean_values)
save_corner_plot(folder_name, samples, symbols_list, mean_values, map_estimate)
save_emcee_summary(folder_name, sampler, symbols_list)
