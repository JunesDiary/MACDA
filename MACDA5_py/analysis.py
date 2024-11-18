from sympy.matrices import Matrix
from sympy.abc import f
import sympy as sp
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.widgets import Slider, Button
import os
import corner
import arviz as az


from createblocks import unitcell
from threesection import get_s11eq_three, get_s21eq_three, get_s31eq_three, get_s41eq_three




def qhdc_creater_plotter_slow(derived_imp):
    #input
    eps_r = 4.0
    f_res = 25 * pow(10, 9)  #GHz
    z_0 = 50
    z_line = z_0/math.sqrt(2)
    
    
    # Define symbols and electrical length for stub and TL
    f = sp.symbols('f')
        
    # chooseodd  invert
    imp_list = [z_0] + derived_imp + derived_imp[::-1][1:] + [z_0]
    #print(imp_list)
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
    
    gamma_e = (a_e + (b_e/z_0) - (c_e*z_0) - d_e ) / (a_e + (b_e/z_0) + (c_e*z_0) + d_e)
    gamma_o = (a_o + (b_o/z_0) - (c_o*z_0) - d_o ) / (a_o + (b_o/z_0) + (c_o*z_0) + d_o)
    t_e = 2 / (a_e + (b_e/z_0) + (c_e*z_0) + d_e)
    t_o = 2 / (a_o + (b_o/z_0) + (c_o*z_0) + d_o)
    
    s11 = (gamma_e + gamma_o) / 2
    s21 = (t_e + t_o) / 2
    s31 = (t_e - t_o) / 2
    s41 = (gamma_e - gamma_o) / 2
  
    # plotting
    
    s11_f = sp.lambdify(f, s11, 'numpy')  
    s21_f = sp.lambdify(f, s21, 'numpy')  
    s31_f = sp.lambdify(f, s31, 'numpy')  
    s41_f = sp.lambdify(f, s41, 'numpy')  
    
    #plot ===============================================================================
    
    freq_Hz = np.linspace(1 * pow(10, 9), 50 * pow(10, 9), 1000)
    # Convert frequency values from Hz to GHz
    freq_GHz = freq_Hz / pow(10, 9)
    
    # Evaluate s11_f for the generated frequency values
    s11_plot_dB = 20 * np.log10(np.abs(s11_f(freq_Hz)))
    s21_plot_dB = 20 * np.log10(np.abs(s21_f(freq_Hz)))
    s31_plot_dB = 20 * np.log10(np.abs(s31_f(freq_Hz)))
    s41_plot_dB = 20 * np.log10(np.abs(s41_f(freq_Hz)))
    
        
    # Plot the function
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(freq_GHz, s11_plot_dB, label='S11')
    ax1.plot(freq_GHz, s21_plot_dB, label='S21')
    ax1.plot(freq_GHz, s31_plot_dB, label='S31')
    ax1.plot(freq_GHz, s41_plot_dB, label='S41')

    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('S Parameter (dB)')
    #ax1.set_title('S Parameters Plot (dB)')
    ax1.grid(True)

    # Print the frequency values
    print(f"Center: {f_res / 1e9:.2f} GHz")

    # Find indices where the absolute difference between s11_plot_dB and -10 is minimum
    closest_to_minus_10_indices = np.argsort(np.abs(np.array(s11_plot_dB) + 20))[:2]

    # Ensure there are two occurrences
    if len(closest_to_minus_10_indices) == 2:
        index_1, index_2 = closest_to_minus_10_indices
        f_cut_1 = round(freq_GHz[index_1], 1)
        f_cut_2 = round(freq_GHz[index_2], 1)

        bw = round(np.abs(f_cut_2 - f_cut_1), 1)
        bw_frac = round(bw / (f_res / 1e9), 3)
        print(f'{f_cut_1} GHz - {f_cut_2} GHz, BW: {bw} GHz ({bw_frac})')
    else:
        print("There are not enough occurrences with values closest to -10.")

    ax1.axvline(x=freq_GHz[index_1], color='m', linestyle='--', linewidth=0.6)
    ax1.axvline(x=freq_GHz[index_2], color='m', linestyle='--', linewidth=0.6)
    ax1.axhline(y=-3, color='y', linestyle='--', linewidth=0.6)

    ax1.legend(title='Lines', loc="best")
    ax1.set_ylim(-40, 0)

    # Create the second figure and plot
    fig2, ax2 = plt.subplots(figsize=(8, 2))
    s21_s31_diff_db = s21_plot_dB - s31_plot_dB

    ax2.plot(freq_GHz, s21_s31_diff_db, label='S21-S31')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S21 - S31 (dB)')
    
    bw_limit = -1.8
    # Find the index of the peak in s21_s31_diff_db
    peak_index = np.argmax(s21_s31_diff_db)

    # Find all indices where the value crosses -1 or is -1
    crossing_indices = []
    for i in range(1, len(s21_s31_diff_db)):
        if (s21_s31_diff_db[i - 1] < -1.8 <= s21_s31_diff_db[i]) or (s21_s31_diff_db[i - 1] >= -1.8 > s21_s31_diff_db[i]):
            crossing_indices.append(i)

    # Include indices where the value is exactly -1
    exact_crossing_indices = [i for i, value in enumerate(s21_s31_diff_db) if value == bw_limit]
    crossing_indices.extend(exact_crossing_indices)
    crossing_indices = sorted(set(crossing_indices))  # Remove duplicates and sort
    crossing_indices = list([crossing_indices[1], crossing_indices[2]])
    temp_freq_list = []
    # Plot vertical lines and labels at crossing points
    for index in crossing_indices:
        ax2.axvline(x=freq_GHz[index], color='m', linestyle='--', linewidth=0.6)
        ax2.text(freq_GHz[index], 0.8, f'{freq_GHz[index]:.1f} GHz', color='navy', ha='center', va='bottom')
        temp_freq_list.append(freq_GHz[index])
    
    temp_freq_list = sorted(temp_freq_list)
    diff_bw_s21_s31 = round((temp_freq_list[1] - temp_freq_list[0]) / (f_res / 1e9), 3)
    
    # Draw horizontal line at -1
    ax2.axhline(y=-1.8, color='r', linestyle='--', linewidth=1, label = 'diff = 1.8/-1.8')
    ax2.axhline(y=1.8, color='r', linestyle='--', linewidth=1)
    #ax2.axhline(y=s21_s31_diff_db[peak_index], color='y', linestyle='--', linewidth=0.6, label = 'peak diff')
    
    # Set title and grid
    print(f'Diff: S21 - S31 ({diff_bw_s21_s31}) offset {round(s21_s31_diff_db[peak_index], 2)} dB')
    ax2.grid(True)
    #ax2.legend(loc="best")
    
    fig3, ax3 = plt.subplots(figsize=(8, 2))
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('S21/S31+3 (dB)')
    ax3.plot(freq_GHz, np.abs(s21_plot_dB+3), label='S21+3')
    ax3.plot(freq_GHz, np.abs(s31_plot_dB+3), label='S31+3')
    ax3.grid(True)
    ax3.legend(loc="best")
    ax3.axhline(y=1, color='y', linestyle='--', linewidth=0.6, label = 'diff = 1')
    ax3.set_ylim(0,2)
    # Return the Figure object
    return fig1, fig2



def impedance_tweaker(mean_values, f_res):
    def qhdc_eq_fetch(derived_imp):
        #input
        z_0 = 50
        
        s11_plot_dB_pre = []
        s21_plot_dB_pre = []
        s31_plot_dB_pre = []
        s41_plot_dB_pre = []

        for f_test in freq_Hz:
            s11_plot_dB_pre.append(get_s11eq_three(*np.append([f_test,z_0, 50], derived_imp)))
            s21_plot_dB_pre.append(get_s21eq_three(*np.append([f_test,z_0, 50], derived_imp)))
            s31_plot_dB_pre.append(get_s31eq_three(*np.append([f_test,z_0, 50], derived_imp)))
            s41_plot_dB_pre.append(get_s41eq_three(*np.append([f_test,z_0, 50], derived_imp)))

        s11_plot_dB = 20 * np.log10(np.abs(s11_plot_dB_pre))
        s21_plot_dB = 20 * np.log10(np.abs(s21_plot_dB_pre))
        s31_plot_dB = 20 * np.log10(np.abs(s31_plot_dB_pre))
        s41_plot_dB = 20 * np.log10(np.abs(s41_plot_dB_pre))
    
        return [s11_plot_dB, s21_plot_dB, s31_plot_dB, s41_plot_dB]
    
    freq_Hz = np.linspace(1 * pow(10, 9), 50 * pow(10, 9), 1000)
    # Convert frequency values from Hz to GHz
    freq_GHz = freq_Hz / pow(10, 9)

    # Get output
    s11_variable, s21_variable, s31_variable, s41_variable = qhdc_eq_fetch(list(mean_values))
    # Calculate the difference
    s21_minus_s31 = s21_variable - s31_variable

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the first four S-parameters
    line_s11, = ax1.plot(freq_GHz, s11_variable, label='S11')
    line_s21, = ax1.plot(freq_GHz, s21_variable, label='S21')
    line_s31, = ax1.plot(freq_GHz, s31_variable, label='S31')
    line_s41, = ax1.plot(freq_GHz, s41_variable, label='S41')
    #ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('S Parameter (dB)')
    ax1.grid(True)
    ax1.legend()

    # Plot the difference in the second subplot
    line_diff, = ax2.plot(freq_GHz, s21_minus_s31, label='S21 - S31', linestyle='--', color='orange')
    ax2.axhline(y=-1.8, color='red', linestyle='--')
    ax2.axhline(y=1.8, color='red', linestyle='--')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S21-S31 (dB)')
    ax2.grid(True)
    ax2.legend()


    # Create slider axes
    axis = plt.axes([0.75, 0.95, 0.14, 0.015])
    axis2 = plt.axes([0.75, 0.85, 0.14, 0.015])
    axis3 = plt.axes([0.75, 0.75, 0.14, 0.015])
    axis4 = plt.axes([0.75, 0.65, 0.14, 0.015])

    # Create sliders
    slider1 = Slider(axis, 'First', mean_values[0] - 25, mean_values[0] + 25, valinit=mean_values[0])
    slider2 = Slider(axis2, "Second", mean_values[1] - 25, mean_values[1] + 25, valinit=mean_values[1])
    slider3 = Slider(axis3, "Third", mean_values[2] - 25, mean_values[2] + 25, valinit=mean_values[2])
    slider4 = Slider(axis4, "Fourth", mean_values[3] - 25, mean_values[3] + 25, valinit=mean_values[3])

    # Create button axes for arrows (adjusted position to avoid overlap)
    button_ax1_up = plt.axes([0.85, 0.95, 0.025, 0.02])
    button_ax1_down = plt.axes([0.75, 0.95, 0.025, 0.02])
    button_ax2_up = plt.axes([0.85, 0.85, 0.025, 0.02])
    button_ax2_down = plt.axes([0.75, 0.85, 0.025, 0.02])
    button_ax3_up = plt.axes([0.85, 0.75, 0.025, 0.02])
    button_ax3_down = plt.axes([0.75, 0.75, 0.025, 0.02])
    button_ax4_up = plt.axes([0.85, 0.65, 0.025, 0.02])
    button_ax4_down = plt.axes([0.75, 0.65, 0.025, 0.02])

    # Create buttons
    button1_up = Button(button_ax1_up, '^', color='lightgoldenrodyellow')
    button1_down = Button(button_ax1_down, 'v', color='lightgoldenrodyellow')
    button2_up = Button(button_ax2_up, '^', color='lightgoldenrodyellow')
    button2_down = Button(button_ax2_down, 'v', color='lightgoldenrodyellow')
    button3_up = Button(button_ax3_up, '^', color='lightgoldenrodyellow')
    button3_down = Button(button_ax3_down, 'v', color='lightgoldenrodyellow')
    button4_up = Button(button_ax4_up, '^', color='lightgoldenrodyellow')
    button4_down = Button(button_ax4_down, 'v', color='lightgoldenrodyellow')


    # Initialize a list to keep track of the frequency label text objects
    label_texts = []

    def update(val):
        # Clear previous labels
        for label in label_texts:
            label.remove()
        label_texts.clear()  # Clear the list

        new_mean_values = [
            slider1.val,
            slider2.val,
            slider3.val,
            slider4.val
        ]
        
        # Get updated output based on new mean values
        s11_variable, s21_variable, s31_variable, s41_variable = qhdc_eq_fetch(new_mean_values)
        
        # Update the plot lines for S-parameters
        line_s11.set_ydata(s11_variable)
        line_s21.set_ydata(s21_variable)
        line_s31.set_ydata(s31_variable)
        line_s41.set_ydata(s41_variable)
        
        # Calculate the difference and update the second plot
        s21_minus_s31 = s21_variable - s31_variable
        line_diff.set_ydata(s21_minus_s31)
        peak_index = np.argmax(s21_minus_s31)

        # Find the crossing indices again
        crossing_indices = []
        for i in range(1, len(s21_minus_s31)):
            if (s21_minus_s31[i - 1] < -1.8 <= s21_minus_s31[i]) or (s21_minus_s31[i - 1] >= -1.8 > s21_minus_s31[i]):
                crossing_indices.append(i)

        exact_crossing_indices = [i for i, value in enumerate(s21_minus_s31) if value == -1.8]
        crossing_indices.extend(exact_crossing_indices)
        crossing_indices = sorted(set(crossing_indices))  # Remove duplicates and sort

        # Plot vertical lines and labels at crossing points
        for index in crossing_indices:
            #ax2.axvline(x=freq_GHz[index], color='m', linestyle='--', linewidth=0.6)
            label = ax2.text(freq_GHz[index], 0.8, f'{freq_GHz[index]:.1f} GHz', color='m', fontsize=14, ha='center', va='bottom')
            label_texts.append(label)  # Keep track of the label

        # Calculate frequency differences if there are enough crossing points
        temp_freq_list = sorted([freq_GHz[i] for i in crossing_indices])
        if len(temp_freq_list) > 2:
            diff_bw_s21_s31 = round((temp_freq_list[2] - temp_freq_list[1]) / (f_res / 1e9), 3)
            ax2.set_title(f'Diff: S21 - S31 ({diff_bw_s21_s31}) offset {round(s21_minus_s31[peak_index], 2)} dB')

        # Redraw the figures
        fig.canvas.draw_idle()

    # Attach the update function to the sliders
    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)
    slider4.on_changed(update)

    # Button functions for sliders
    def adjust_slider(slider, delta):
        slider.set_val(slider.val + delta)
        update(None)

    least_count = 0.2
    button1_up.on_clicked(lambda event: adjust_slider(slider1, least_count))
    button1_down.on_clicked(lambda event: adjust_slider(slider1, -least_count))
    button2_up.on_clicked(lambda event: adjust_slider(slider2, least_count))
    button2_down.on_clicked(lambda event: adjust_slider(slider2, -least_count))
    button3_up.on_clicked(lambda event: adjust_slider(slider3, least_count))
    button3_down.on_clicked(lambda event: adjust_slider(slider3, -least_count))
    button4_up.on_clicked(lambda event: adjust_slider(slider4, least_count))
    button4_down.on_clicked(lambda event: adjust_slider(slider4, -least_count))

    # Create button axes for the reset button
    reset_button_ax = plt.axes([0.05, 0.05, 0.14, 0.05])  # Adjust position as needed
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow')

    # Define the reset function
    def reset(event):
        slider1.set_val(mean_values[0])
        slider2.set_val(mean_values[1])
        slider3.set_val(mean_values[2])
        slider4.set_val(mean_values[3])
        update(None)  # Call update to refresh the plot with the reset values

    # Connect the reset button to the reset function
    reset_button.on_clicked(reset)

    plt.show()



def save_parameter_timeseries(samples_whole, filename, symbols_list, map_estimate, mean_values):
    # Create the output folder if it doesn't exist
    folder_name_conv = os.path.join(filename, 'convergencetest')
    # Create the folder if it doesn't exist
    os.makedirs(folder_name_conv, exist_ok=True)

    # Extract the entire sample chain
    
    ndim = samples_whole.shape[2]  # Number of parameters

    # Initialize plot
    fig_samples_all, axes = plt.subplots(ndim, figsize=(9, 8), sharex=True)
    fs = 17  # Font size

    # Set font properties
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'DejaVu Serif',
        'font.size': fs,
        'axes.titlesize': fs,
        'axes.labelsize': fs,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'legend.fontsize': fs
    })

    # Plot each parameter trace
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples_whole[:, :, i], "k", alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, len(samples_whole))
        ax.set_ylabel(symbols_list[i], fontsize=fs)
        ax.yaxis.set_label_coords(-0.1, 0.5)

        # Add horizontal lines for actual values, MAP, and mean
        ax.axhline(y=map_estimate[i], color='r', linestyle='-', linewidth=2, label='MAP')
        ax.axhline(y=mean_values[i], color='y', linestyle='--', linewidth=2, label='Mean')

        # Increase tick label sizes and reduce the number of x-ticks
        ax.tick_params(axis='both', which='major', labelsize=fs)
        xticks = ax.get_xticks()
        reduced_xticks = xticks[::len(xticks) // 4]
        ax.set_xticks(reduced_xticks)

    # Set xlabel for the last subplot
    axes[-1].set_xlabel("Step number", fontsize=fs)

    # Add legend in the last subplot
    axes[-1].legend(loc='upper right', fontsize=fs)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(folder_name_conv, "parameter_timeseries.png")
    plt.savefig(output_path)

    # Close the plot to free memory
    plt.close(fig_samples_all)
    print(f"Parameter traces saved to {output_path}")


def save_corner_plot(filename, samples, symbols_list, mean_values, map_estimate):
    tick_frequency=2
    folder_name_conv = os.path.join(filename, 'convergencetest')
    fs = 18  # Font size
    plt.rcParams.update({
        'font.size': fs,
        'axes.titlesize': fs,
        'axes.labelsize': fs,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'legend.fontsize': fs,
        'font.serif': ['DejaVu Serif']
    })

    # Create the corner plot
    figure_corner = corner.corner(samples, labels=symbols_list, show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": fs})
    ndim = samples.shape[1]
    axes = figure_corner.axes

    # Loop over the axes to set tick frequency and format labels
    for i, ax in enumerate(axes):
        # Set reduced tick frequency for x and y ticks
        ax.set_xticks(ax.get_xticks()[::tick_frequency])
        ax.set_yticks(ax.get_yticks()[::tick_frequency])

        # Format x-axis and y-axis labels with subscripts
        if i % ndim == 0:  # Only for the first column
            ax.set_ylabel(f"$\\mathrm{{zv}}_{{{i // ndim + 1}}}$", fontsize=fs)
        if i >= ndim * (ndim - 1):  # Only for the last row
            ax.set_xlabel(f"$\\mathrm{{zv}}_{{{i % ndim + 1}}}$", fontsize=fs)

    # Add mean, MAP, and actual value lines on the diagonal
    for i in range(ndim):
        ax = axes[i * ndim + i]
        ax.axvline(mean_values[i], color="g", linestyle="--", label="Mean")
        ax.axvline(map_estimate[i], color="r", linestyle="--", label="MAP Estimate")

    # Add filled contours and markers on the off-diagonal scatter plots
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi * ndim + xi]
            hist, xedges, yedges = np.histogram2d(samples[:, xi], samples[:, yi], bins=30)
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

            ax.contourf(X, Y, hist.T, levels=15, cmap='viridis', alpha=0.8)

            # Vertical and horizontal lines for mean, MAP, and actual values
            ax.axvline(mean_values[xi], color="g", linestyle="--")
            ax.axvline(map_estimate[xi], color="r", linestyle="--")
            ax.axhline(mean_values[yi], color="g", linestyle="--")
            ax.axhline(map_estimate[yi], color="r", linestyle="--")

            # Mean, MAP, and actual value markers
            ax.plot(mean_values[xi], mean_values[yi], "sg")
            ax.plot(map_estimate[xi], map_estimate[yi], "sr")

    # Save the plot
    output_path = os.path.join(folder_name_conv, "corner_plot.png")
    figure_corner.savefig(output_path, bbox_inches='tight', dpi=300)

    # Close the plot to free memory
    plt.close(figure_corner)
    print(f"Corner plot saved to {output_path}")

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    if norm:
        acf /= acf[0]

    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def save_emcee_summary(filename, sampler, symbols_list):
    folder_name_analysis = os.path.join(filename, 'statistic_analysis')
    os.makedirs(folder_name_analysis, exist_ok=True)

    # Convert emcee sampler to ArviZ InferenceData
    idata1 = az.from_emcee(sampler, var_names=symbols_list)

    # Generate summary
    summary = az.summary(idata1)
    
    # Save summary to readme.txt
    readme_path = os.path.join(filename, "readme.txt")
    with open(readme_path, "a") as file:
        file.write("\nConvergence Tests Summary\n")
        file.write("=================\n")
        file.write(summary.to_string())
    
    # Error bar plot
    symbols_str = [str(symbol) for symbol in symbols_list]
    fig, ax = plt.subplots()
    ax.errorbar(symbols_str, 
                np.array(list(summary['mean'])),
                yerr=np.array(list(summary['sd'])), 
                fmt='o', 
                color='cyan', 
                ecolor='orange', 
                elinewidth=2, 
                capsize=5)
    
    ax.set_ylabel('Impedance Mean Values (ohm)')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plot_path = os.path.join(folder_name_analysis, "impedance_sdvar_plot.png")
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Histogram plot of posterior distributions
    fig, ax = plt.subplots()
    for i in range(4):
        chain = sampler.get_chain()[:, :, i].T
        ax.hist(chain.flatten(), bins=150, label=str(symbols_list[i]))
    ax.set_yticks([])
    ax.set_xlabel("Impedance (ohm)")
    ax.set_ylabel("Density (counts)")
    ax.legend()
    plot_path = os.path.join(folder_name_analysis, "impedance_post_dist.png")
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Autocorrelation plot
    chain = sampler.get_chain()[:, :, 0].T
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(chain[:, :n])
        new[i] = autocorr_new(chain[:, :n])

    fig, ax = plt.subplots()
    ax.loglog(N, gw2010, "o-", label="G&W 2010")
    ax.loglog(N, new, "o-", label="New")
    ax.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    ax.set_xlabel("Number of samples, $N$")
    ax.set_ylabel(r"$\tau$ estimates")
    ax.legend()

    # Finding and marking intersection point
    difference = gw2010 - (N / 50.0)
    indices = np.where(np.diff(np.sign(difference)))[0]
    if len(indices) > 0:
        idx = indices[0]
        n_intersect = (N[idx] + N[idx + 1]) / 2
        ax.axvline(n_intersect, color='orange', linestyle='--', label=f'{n_intersect:.0f} steps')
    
    plot_path = os.path.join(folder_name_analysis, "autocorr_estimates.png")
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Summary saved to {readme_path}, plots saved in {folder_name_analysis}")

  




