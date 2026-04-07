import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def plot_psu_distribution(psu_data: np.ndarray, new_psu_value: float, save_path: str, x_min=0, x_max=1):
    """Plot the new patient uncertainty value (PSU) wrt PSU distribution.

    Args:
        psu_data (np.ndarray): an array of PSU value from the test population
        new_psu_value (float): the new PSU value to plot
        save_path (str): path to save the plot
        x_min (int, optional): minimum x-axis value. Defaults to 0.
        x_max (int, optional): maximum x-axis value. Defaults to 1.
    """
    # Estimate PDF using kernel density estimation
    kde = stats.gaussian_kde(psu_data)

    # Create a range for plotting
    if x_min is None or x_max is None:
        x_min, x_max = min(psu_data) - (max(psu_data) - min(psu_data)) * 0.1, max(psu_data) + (max(psu_data) - min(psu_data)) * 0.1
    x_plot = np.linspace(x_min, x_max, 500)
    pdf_values = kde(x_plot)

    # Calculate percentile
    percentile = stats.percentileofscore(psu_data, new_psu_value, kind='rank')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot the filled area under the curve
    ax.fill_between(x_plot, pdf_values, color='#D5EDD2', alpha=0.8)

    # Plot the curve itself
    ax.plot(x_plot, pdf_values, color='#76BF6A', linewidth=2)

    # Add the current_x marker (person icon)
    # Find the y-value of the KDE at current_x
    current_x_pdf_value = kde(new_psu_value)[0]
    
    ax.plot(
    new_psu_value,
    current_x_pdf_value,
    marker='o',
    markersize=12,
    color='#2BBAB7',
    linestyle='None',
    zorder=5
    )

    # Vertical line from dot to x-axis
    ax.vlines(
        x=new_psu_value,
        ymin=0,
        ymax=current_x_pdf_value,
        colors='#2BBAB7',
        linestyles='dashed',
        linewidth=2,
        zorder=4
    )

    ax.annotate("", xy=(new_psu_value, current_x_pdf_value), 
                xytext=(new_psu_value, current_x_pdf_value * 1.5),
                # arrowprops=dict(facecolor='#2BBAB7', edgecolor='none', shrink=0.05, width=5, headwidth=15),
                fontsize=10, ha='center')

    # Customize axes
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_yticklabels([]) # Remove y-axis labels
    ax.set_frame_on(False) # Remove the frame around the plot

    # Set x-axis labels to min and max of data
    ax.set_xticks([np.floor(min(psu_data)), new_psu_value, np.ceil(max(psu_data))])
    ax.set_xticklabels([f'{int(np.floor(min(psu_data)))}', f'{new_psu_value:.2f}', f'{int(np.ceil(max(psu_data)))}'], fontsize=12, color='#666666')

    # Title with percentage
    ax.text(0.5, 1.05, f"{int(100 - percentile)} % of patients have higher uncertainty",
            transform=ax.transAxes, fontsize=20, color='#666666',
            ha='center', va='bottom')
    # ax.text(0.65, 1.05, f"{int(percentile)}", transform=ax.transAxes, fontsize=28, color='#76BF6A', ha='center', va='bottom')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()





def plot_psc_distribution(psc_data: np.ndarray, new_psc_value: float, save_path: str, x_min=0, x_max=1):
    """Plot the new patient uncertainty value (PSU) wrt PSU distribution.

    Args:
        psc_data (np.ndarray): an array of PSC value from the test population
        new_psc_value (float): the new PSC value to plot
        save_path (str): path to save the plot
        x_min (int, optional): minimum x-axis value. Defaults to 0.
        x_max (int, optional): maximum x-axis value. Defaults to 1.
    """
    # Estimate PDF using kernel density estimation
    kde = stats.gaussian_kde(psc_data)

    # Create a range for plotting
    if x_min is None or x_max is None:
        x_min, x_max = min(psc_data) - (max(psc_data) - min(psc_data)) * 0.1, max(psc_data) + (max(psc_data) - min(psc_data)) * 0.1
    x_plot = np.linspace(x_min, x_max, 500)
    pdf_values = kde(x_plot)

    # Calculate percentile
    percentile = stats.percentileofscore(psc_data, new_psc_value, kind='rank')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot the filled area under the curve
    ax.fill_between(x_plot, pdf_values, color='#D5EDD2', alpha=0.8)

    # Plot the curve itself
    ax.plot(x_plot, pdf_values, color='#76BF6A', linewidth=2)

    # Add the current_x marker (person icon)
    # Find the y-value of the KDE at current_x
    current_x_pdf_value = kde(new_psc_value)[0]
    
    ax.plot(
    new_psc_value,
    current_x_pdf_value,
    marker='o',
    markersize=12,
    color='#2BBAB7',
    linestyle='None',
    zorder=5
    )

    # Vertical line from dot to x-axis
    ax.vlines(
        x=new_psc_value,
        ymin=0,
        ymax=current_x_pdf_value,
        colors='#2BBAB7',
        linestyles='dashed',
        linewidth=2,
        zorder=4
    )

    ax.annotate("", xy=(new_psc_value, current_x_pdf_value), 
                xytext=(new_psc_value, current_x_pdf_value * 1.5),
                # arrowprops=dict(facecolor='#2BBAB7', edgecolor='none', shrink=0.05, width=5, headwidth=15),
                fontsize=10, ha='center')

    # Customize axes
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_yticklabels([]) # Remove y-axis labels
    ax.set_frame_on(False) # Remove the frame around the plot

    # Set x-axis labels to min and max of data
    ax.set_xticks([np.floor(min(psc_data)), new_psc_value, np.ceil(max(psc_data))])
    ax.set_xticklabels([f'{int(np.floor(min(psc_data)))}', f'{new_psc_value:.2f}', f'{int(np.ceil(max(psc_data)))}'], fontsize=12, color='#666666')

    # Title with percentage
    ax.text(0.5, 1.05, f"{int(percentile)} % of patients have lower certainty",
            transform=ax.transAxes, fontsize=20, color='#666666',
            ha='center', va='bottom')
    # ax.text(0.65, 1.05, f"{int(percentile)}", transform=ax.transAxes, fontsize=28, color='#76BF6A', ha='center', va='bottom')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()





# base_path = "/home/fede/SMSC_test_copy/processed"
# files = glob(os.path.join(base_path, '*', '*', '*', 'patient_uncs.csv'))

# rows = []

# for file in files:
#     split = file.split('/')
#     patient = split[-3]
#     visit = split[-2]

#     df = pd.read_csv(file)

#     # take PSU from first row
#     psu = df.loc[0, 'PSU']
#     psc = 1 - psu

#     rows.append({
#         'patient': patient,
#         'visit': visit,
#         'PSU': psu,
#         'PSC': psc
#     })

# # create final dataframe
# psu_data = pd.DataFrame(rows)

# # save csv
# psu_data.to_csv("PSU_data.csv", index=False)


base_path = "/home/federicospagnolo/storage/groups/think/Federico/Report_generation_update"
psu_data = pd.read_csv(os.path.join(base_path, "PSU_data.csv"))

pat_uncs_value = 0.07

plot_psu_distribution(
                    psu_data=np.array(psu_data.loc[:, 'PSU']),
                    new_psu_value=pat_uncs_value,
                    save_path=os.path.join(base_path, f"patient_uncertainty_distribution.png")
                )

pat_cert_value = 0.93

plot_psc_distribution(
                    psc_data=np.array(psu_data.loc[:, 'PSC']),
                    new_psc_value=pat_cert_value,
                    save_path=os.path.join(base_path, f"patient_certainty_distribution.png")
                )