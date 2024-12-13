import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq
# 
target_complexity = 4000 #TEST
L = 3600
W = 500
input_params = {
    "coordinates": [(L*0.5-144, W*0.5+18), (L*0.5+144, W*0.5-18)],
            "inflow_speed":5,
            "domain_length": L,
            "domain_width":W,
            }

print(f'check coordinates: {input_params["coordinates"]}')

        
rootfolder = f"/data0/nn_adapt/output/Dec2024_gnn_test/"

## UNIFORM --------------------------------------------------------------

subfolder = rootfolder+f"target_4000/offset/domain_L{str(L)}W{str(W)}/uniform"

sim_uniform = Simulation(Model,subfolder, parameters = input_params)

sim_uniform.params["adaptor_method"] = 'uniform'
sim_uniform.params["indicator_method"] = "none"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')


sim_uniform.params['miniter'] = 2
sim_uniform.params['maxiter'] = 3


logging.info(sim_uniform.params)
sim_uniform.run_simulation()

# # # BASE - HESSIAN --------------------------------------------------------------
subfolder = rootfolder+f"target_4000/offset/domain_L{str(L)}W{str(W)}//hessian"

sim_uniform = Simulation(Model,subfolder, parameters = input_params)
sim_uniform.params["adaptor_method"] = 'hessian'
sim_uniform.params["indicator_method"] = "none"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

sim_uniform.params['miniter'] = 3
sim_uniform.params['maxiter'] = 14
sim_uniform.params["target"]= target_complexity

logging.info(sim_uniform.params)
sim_uniform.run_simulation()


# # BASE - GO ISOTROPIC --------------------------------------------------------------
subfolder = rootfolder+f"target_4000/offset/domain_L{str(L)}W{str(W)}//goal_based"

sim_uniform = Simulation(Model, subfolder, parameters = input_params)
sim_uniform.params["adaptor_method"] = 'isotropic'
sim_uniform.params["indicator_method"] = "none"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

sim_uniform.params['miniter'] = 3
sim_uniform.params['maxiter'] = 14
sim_uniform.params["target"]= target_complexity

logging.info(sim_uniform.params)
sim_uniform.run_simulation()

# # BASE - GO ANISOTROPIC --------------------------------------------------------------

subfolder = rootfolder+f"target_4000/offset/domain_L{str(L)}W{str(W)}//goal_based"

sim_uniform = Simulation(Model, subfolder, parameters = input_params)
sim_uniform.params["adaptor_method"] = 'anisotropic'
sim_uniform.params["indicator_method"] = "none"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

sim_uniform.params['miniter'] = 3
sim_uniform.params['maxiter'] = 14
sim_uniform.params["target"]= target_complexity

logging.info(sim_uniform.params)
sim_uniform.run_simulation()

# ## MLP - GO ANISOTROPIC -  - ------------------------------------------------------------
subfolder = rootfolder+f"target_4000/offset/domain_L{str(L)}W{str(W)}//ml_mlp"

sim_uniform = Simulation(Model,subfolder, parameters = input_params)
sim_uniform.params["adaptor_method"] = 'anisotropic'
sim_uniform.params["indicator_method"] = "mlp"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

sim_uniform.params['miniter'] = 3
sim_uniform.params['maxiter'] = 14
sim_uniform.params["target"]= target_complexity

logging.info(sim_uniform.params)
sim_uniform.run_simulation()


# ## GNN ALL- GO ANISOTROPIC -  - ------------------------------------------------------------
subfolder = rootfolder+f"target_4000/offset/base/ml_gnn"

sim_uniform = Simulation(Model,subfolder, parameters = input_params)
sim_uniform.params["adaptor_method"] = 'anisotropic'
sim_uniform.params["indicator_method"] = "gnn"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

sim_uniform.params['miniter'] = 3
sim_uniform.params['maxiter'] = 14
sim_uniform.params["target"]= target_complexity

logging.info(sim_uniform.params)
sim_uniform.run_simulation()

import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import os

def plot_cpu_vs_dof(plot_dict, show=True, json_path="."):
    """
    Plots CPU time versus Degrees of Freedom (DoF) for different methods stored in plot_dict.

    This function generates a logarithmic plot where the x-axis represents the degrees of freedom (DoF) 
    and the y-axis represents the CPU time. The plot visualizes the relationship between these two quantities 
    for multiple methods stored in `plot_dict`. Each method is represented by a different color and linestyle. 
    The plot is saved as 'cpu_vs_dof.jpg' in the specified `json_path`.

    Args:
        plot_dict (dict): A dictionary where each key represents a method, and each value contains plot data 
                          such as 'dof_list', 'ftime_list', and plotting style parameters (e.g., 'color', 'linestyle').
        show (bool): If True, the plot will be displayed after being generated. Defaults to True.
        json_path (str): The path where the plot image will be saved. Defaults to the current directory.

    Returns:
        None
    """
    
    # Set up figure
    fig, axes = plt.subplots()

    # Extract metadata from the first item in plot_dict (assuming all methods have consistent metadata)
    first_method = next(iter(plot_dict.values()))
    dof_name = first_method['dof_name']
    dof_unit = first_method['dof_unit']
    
    # Initialize variables for plot limits
    dof_min, dof_max = float('inf'), float('-inf')
    cpu_min, cpu_max = float('inf'), float('-inf')

    # Plot the data for each method
    for key, value in plot_dict.items():
        dof_min = min(dof_min, *value['dof_list'])
        dof_max = max(dof_max, *value['dof_list'])
        cpu_min = min(cpu_min, *value['ftime_list'])
        cpu_max = max(cpu_max, *value['ftime_list'])
        
        axes.loglog(value['dof_list'], value['ftime_list'], '.', label=key, 
                    color=value['color'], linestyle=value['linestyle'])

    # Set labels and grid
    axes.set_xlabel(f"{dof_name} ($\mathrm{{{dof_unit}}}$)")
    axes.set_ylabel(r"CPU time ($\mathrm{s}$)")
    axes.grid(True, which="both")

    # Adjust layout, add legend, and save the figure
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(json_path, 'cpu_vs_dof.jpg'))

    # Display the plot if requested
    if show:
        plt.show()


import os

def create_logfile_list(rootpath, end_pattern=".txt"):
    """
    Searches for all files with a specified extension in the given directory and its subdirectories.

    This function walks through all directories starting from `rootpath` and finds files that match the 
    provided `end_pattern` (default is `.txt`). It returns a list of the file paths that match the pattern.

    Args:
        rootpath (str): The root directory to start searching from.
        end_pattern (str): The file extension pattern to match. Defaults to '.txt'.

    Returns:
        list: A list of full file paths that match the given extension.
    """
    return [os.path.join(root, file) 
            for root, dirs, files in os.walk(rootpath) 
            for file in files 
            if file.endswith(end_pattern)]

