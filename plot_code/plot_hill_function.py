#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# Define Hill function with n = 0.5
def hill_function(x, K=2.0, n=0.8):
    return 0.8*x**n / (K**n + x**n)

#%%
# Generate x values and compute Hill function
x = np.linspace(0.01, 10, 500)
y = hill_function(x)

# Plot without background or axes
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
ax.plot(x, y, linewidth=2)
plt.ylim(0,1)
plt.axhline(0.8)
# Remove background and axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('none')
fig.patch.set_facecolor('none')

# # Save as SVG
svg_path = "/home/mzo5929/Keerthana/grnInference/plots/figure_dummy_plot/hill_function_n_0.8.svg"
plt.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def calc_pi_on(k_on, k_off):
    return k_on / (k_on + k_off)

k_on_range = [0.01]
pi_on_no_reg = np.linspace(0.002, 0.4, 10)
regulator_range = np.linspace(0.0, 10, 100)  # smoother curves

for k_on in k_on_range:
    plt.figure(figsize=(10, 8))  # new figure for each k_on
    for pi_on in pi_on_no_reg:
        k_off = k_on / pi_on - k_on
        k_on_reg = k_on*regulator_range
        eff_pi_on = calc_pi_on(k_on_reg, k_off)
        plt.plot(regulator_range, eff_pi_on/pi_on, linewidth=1.0,
                 label=f'π_on₀ = {pi_on:.2f}')
    
    plt.legend(loc='upper right', fontsize=6, frameon=False)
    plt.xlabel('Regulator concentration')
    plt.ylabel(r'Ratio of Effective $\pi_{on}$ to $\pi_{on0}$')
    plt.title(rf'Ratio of Effective $\pi_{{on}}$ to $\pi_{{on_0}}$ vs Regulator (k_on = {k_on:.2f})')

    plt.tight_layout()
    plt.show()



    