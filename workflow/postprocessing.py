import numpy as np
import matplotlib.pyplot as plt
from mpltools import annotation
import firedrake as fd
import json
import os

def export_plot_params_json(filepath, json_path):
    """
    Extracts and processes plot parameters from a specified input file and exports them into a JSON file.

    This function reads through the provided input file, extracts specific simulation parameters related to the 
    "adaptor method", degrees of freedom (DoF) for velocity and elevation, quantities of interest (QoI), and time 
    taken for various solver steps. It then stores these parameters in a structured format and writes them to a JSON file.

    The following parameters are extracted:
        - Method of adaptation (e.g., 'uniform')
        - Quantity of Interest (QoI) values over time
        - Degrees of freedom (DoF) for velocity and elevation
        - Time taken for forward solving, adaptation, and fixed point iterations

    If the JSON file specified by `json_path` does not already exist, a new one will be created. If it exists, the 
    extracted parameters will be appended to the file, with the `filepath` as the key for the new entry.

    Args:
        filepath (str): Path to the input file from which the parameters will be extracted.
        json_path (str): Directory where the JSON file containing the plot parameters will be saved.

    Returns:
        None

    Example:
        export_plot_params_json('simulation_output.txt', '/path/to/output/')
    """
    if not json_path:
        json_path = os.getcwd()
        print(f'Plot will be saved in: {json_path}')

    plot_params = {
        "qoi": [],
        "dof": [],
        "qoi_name": "power output",
        "qoi_unit": "MW",
        "dof_name": "DoF",
        "dof_unit": "count",
        "ftime": [],
        "fiter": 0,
    }
    dof_list=[]
    qoi_list=[]
    itime_list=[]

    # Initialize placeholders for extracted values
    qoi = 0
    adapt_method = ""
    atime = ftime = fiter = None
    elev_dof = []
    vel_dof = []
    stime = []

    # process file
    skip_n_lines = 0
    # open file
    with open(filepath) as file:
        # if need to skip header lines
        for _snl in range(skip_n_lines):
            file.readline()
        line = None

        elev_dof = vel_dof = qoi = stime = atime = ftime = fiter = None
        # Read each line that is not blank
        while line not in ['\n', '\r\n', '']:
            line = file.readline()

            if "Input parameters" in line:
                ainfo = line[line.find("'adaptor_method'"):line.find("'fix_boundary'")].split(':')
                adapt_method  = str(ainfo[1].strip().split('_')[-1].strip("',"))
                iinfo = line[line.find("'indicator_method'"):line.find("'adaptor_method'")].split(':')
                indicator_method  = str(iinfo[1].strip().split('_')[-1].strip("',"))
            elif "Number of 2D elevation DOFs:" in line:
                elev_dof = int(line.split(":")[-1].strip())
            elif "Number of 2D velocity DOFs:" in line:
                vel_dof = int(line.split(":")[-1].strip())
            elif "qoi:" in line:
                qoi = float(line.split(":")[-1].strip())
            elif "solve - forward solve, time taken: " in line:
                stime = float(line.split(":")[-1].strip())
            elif "adaptor - adaptor, time taken: " in line:
                atime = float(line.split(":")[-1].strip())
            elif "fixed point iteration loop, time taken: " in line:
                ftime = float(line.split(":")[-1].strip())
            elif "fixed point iterator total iterations:" in line:
                fiter = int(line.split(":")[-1].strip())

            # Collect QoI and DoF once all relevant data is found
            if qoi and elev_dof and vel_dof:
                qoi_list.append(qoi)
                # for anisotropic dwr - there are 2 solves, the second is the correct mesh
                dof_list.append(elev_dof[-1] + vel_dof[-1])
                elev_dof = []
                vel_dof = []
                qoi=None

            # Add times to the list
            if stime and atime:
                # for anisotropic dwr - there are 2 solves, the first is base solve
                itime_list.append(stime[0] + atime)
                stime = []
                atime = 0
    
    # Update plot_params with extracted values
    plot_params.update({
        "method": adapt_method if indicator_method == 'none' else indicator_method,
        "qoi": qoi_list,
        "dof": dof_list,
        "fiter": fiter,
        "qoi_name": "power output",
        "qoi_unit": "MW",
        "dof_name": "DoF",
        "dof_unit": "count",
        "ftime": itime_list if adapt_method == "uniform" else ftime,
    })
    
    # Write or update JSON file
    json_file_path = os.path.join(json_path, 'plot_parameters.json')
    if os.path.exists(json_file_path):
        # get existing entries from current json file
        with open(json_file_path, 'r+') as fp:
            params_dict = json.load(fp)
        # append new parameterst
        params_dict[filepath] = plot_params
        # save updated dictionary to json
        with open(json_file_path, 'w') as fp:
            json.dump(params_dict, fp, indent=4)
    else:
        print("no path")
        with open(json_file_path, 'w') as fp:
            # initialize dictionary with current parameters
            json.dump({filepath:plot_params}, fp,indent=4)


def create_plot_dictionary(json_path):
    """
    *from ChatGPT*
    Creates a dictionary of plot parameters based on data from a JSON file.

    This function loads plot parameter data from a JSON file located at `json_path/plot_parameters.json`. 
    It processes the data and organizes it into a master dictionary (`plot_dict`) where each entry represents 
    a specific method of adaptation (e.g., 'uniform', 'hessian', 'isotropic', 'anisotropic'). For each method, 
    a corresponding plot template is created and populated with relevant data such as quantities of interest (QoI), 
    degrees of freedom (DoF), time taken for each solver step, and iteration counts.

    The function supports both a default template for 'uniform' methods and dynamic templates for other methods 
    (e.g., 'hessian', 'isotropic', 'anisotropic'), which are customized with method-specific attributes like 
    `color`, `linestyle`, and data.

    Args:
        json_path (str): The directory path containing the 'plot_parameters.json' file, which holds the plot 
                         parameter data.

    Returns:
        dict: A master dictionary (`plot_dict`) containing plot data for each adaptation method. Each entry in 
              the dictionary is keyed by the method name (e.g., 'uniform', 'hessian', etc.) and contains a 
              dictionary with keys such as 'color', 'linestyle', 'qoi_list', 'dof_list', 'ftime_list', 
              'fiter_list', and other associated plot data.

    Example:
        plot_dict = create_plot_dictionary('/path/to/json')
    """

    try:
        with open(os.path.join(json_path, 'plot_parameters.json'), 'r') as fp:
            plot_params = json.load(fp)
    except BaseException as error:
        print(f'An exception occurred: {error}')
        
    # Global master dictionary
    plot_dict = {}

    # template for each dictionary entry
    def create_dict(instance_name, custom_values=None):
        # Default template values
        template = {
            "color":"blue",
            "linestyle":"--",
            'qoi_list': [],
            'dof_list': [],
            'ftime_list': [],
            'fiter_list': [], 
            'qoi_name': None, 
            'qoi_unit': None, 
            'dof_name': None, 
            'dof_unit': None,
        }
        # If custom values are provided, update the template with them
        if custom_values:
            template.update(custom_values)
        
        # Add the generated instance to the master dictionary using the instance name as key
        plot_dict[instance_name] = template
        

    # Method-color map for non-uniform methods
    method_colors = {
        'hessian': {"color": "brown", "linestyle": ":"},
        'isotropic': {"color": "green", "linestyle": ":"},
        'anisotropic': {"color": "orange", "linestyle": "--"},
        'mlp':{"color": "red", "linestyle": "-"},
        'gnn':{"color": "pink", "linestyle": "-"},
    }

    # Populate the plot_dict from the plot_params
    for params in plot_params.values():
        method = params["method"]
        
        if method == 'uniform':
            create_dict('uniform', {"color": "blue", "linestyle": "--"})
            
            # Append the values to the lists
            plot_dict[method]['qoi_list'] += params['qoi'][1:]
            plot_dict[method]['dof_list'] += params['dof'][1:]
            plot_dict[method]['ftime_list'] += params['ftime'][1:]
            plot_dict[method]['fiter_list']= [1]*params['fiter']
        else:
            # For other methods, initialize them if not already done
            if method not in plot_dict:
                create_dict(method, method_colors.get(method, {"color": "gray", "linestyle": "-"}))
            
            # Append the latest values for each key
            plot_dict[method]['qoi_list'].append(params['qoi'][-1])
            plot_dict[method]['dof_list'].append(params['dof'][-1])
            plot_dict[method]['ftime_list'].append(params['ftime'])
            plot_dict[method]['fiter_list'].append(params['fiter'])

        # Update parameters if they haven't been set yet
        for key in ['qoi_name', 'qoi_unit', 'dof_name', 'dof_unit']:
            if plot_dict[method][key] is None:
                plot_dict[method][key] = params[key]

    return plot_dict


def plot_qoi_vs_dof(plot_dict, show=True, json_path=None):
    """
    *modified from ChatGPT*

    Plots a Quantity of Interest (QoI) vs Degrees of Freedom (DoF) for different methods stored in plot_dict.

    This function generates a semi-logarithmic plot where the x-axis represents the degrees of freedom (DoF) 
    and the y-axis represents the quantity of interest (QoI). The plot visualizes the relationship between these 
    two quantities for multiple methods stored in `plot_dict`. Each method is represented by a different color and 
    linestyle. The plot is saved as 'qoi_vs_dof.jpg' in the specified `json_path`.

    Args:
        plot_dict (dict): A dictionary where each key represents a method, and each value contains plot data 
                          such as 'dof_list', 'qoi_list', and plotting style parameters (e.g., 'color', 'linestyle').
        show (bool): If True, the plot will be displayed after being generated. Defaults to True.
        json_path (str): The path where the plot image will be saved. Defaults to the current directory.

    Returns:
        None
    """

    if not json_path:
        json_path = os.getcwd()
    
    print(f'Plot will be saved in: {json_path}')

    # Set up figure
    fig, axes = plt.subplots()
    
    # Extract metadata from the first item in plot_dict (assuming all methods have consistent metadata)
    first_method = next(iter(plot_dict.values()))
    qoi_name = first_method['qoi_name']
    qoi_unit = first_method['qoi_unit']
    dof_name = first_method['dof_name']
    dof_unit = first_method['dof_unit']

    # Initialize variables for plot limits
    dof_min, dof_max = _set_min_max()
    qoi_min, qoi_max  = _set_min_max()

    # Plot the data for each method
    for key, value in plot_dict.items():
        dof_min, dof_max = _set_min_max([dof_min, dof_max],value['dof_list'])
        qoi_min, qoi_max = _set_min_max([qoi_min, qoi_max],value['qoi_list'])
        _dofs, _qois = joint_sort_lists([value['dof_list'],value['qoi_list']],0)
        axes.loglog(_dofs, _qois, '.', label=key, 
                    color=value['color'], linestyle=value['linestyle'])
    
    # Set labels and grid
    axes.set_xlabel(f"{dof_name} ($\mathrm{{{dof_unit}}}$)")
    axes.set_ylabel(f"{qoi_name} ($\mathrm{{{qoi_unit}}}$)")
    axes.grid(True)

    # Adjust layout, add legend, and save the figure
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(json_path, 'qoi_vs_dof.jpg'))

    #Display the plot if requested
    if show:
        plt.show


def _set_min_max(minmax = None, update_values = None):
    """
    Updates a [min,max] list given a list of update values to consider, retaining
    the min and max values only based on updated range

    Args:
        minmax (list): two digit list containing min, max value [min,max]
        update_values (list): list containing additional values to consider in range

    Returns:
        minmax (list): updated [min,max]
    """
    # define default
    if minmax is None:
        return [float('-inf'), float('inf')]
    
    if update_values is None:
        return minmax

    return [min(minmax[0], *update_values),max(minmax[1], *update_values)]


def plot_cpu_vs_dof(plot_dict, show=True, json_path=None):
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
    
    if not json_path:
        json_path = os.getcwd()
    
    print(f'Plot will be saved in: {json_path}')

    # Set up figure
    fig, axes = plt.subplots()

    # Extract metadata from the first item in plot_dict (assuming all methods have consistent metadata)
    first_method = next(iter(plot_dict.values()))
    dof_name = first_method['dof_name']
    dof_unit = first_method['dof_unit']
    
    # Initialize variables for plot limits
    dof_min, dof_max = _set_min_max()
    cpu_min, cpu_max = _set_min_max()

    # Plot the data for each method
    for key, value in plot_dict.items():
        print(value['ftime_list'])
        dof_min, dof_max = _set_min_max([dof_min, dof_max],value['dof_list'])
        cpu_min, cpu_max = _set_min_max([cpu_min, cpu_max],value['ftime_list'])
        _dofs, _cpus = joint_sort_lists([value['dof_list'],value['ftime_list']],0)
        axes.loglog(_dofs, _cpus, '.', label=key, 
                    color=value['color'], linestyle=value['linestyle'])

    # Set labels and grid
    axes.set_xlabel(f"{dof_name} ($\mathrm{{{dof_unit}}}$)")
    axes.set_ylabel(f"CPU time ($\mathrm{{s}}$)")
    axes.grid(True, which="both")

    # Adjust layout, add legend, and save the figure
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(json_path, 'cpu_vs_dof.jpg'))

    # Display the plot if requested
    if show:
        plt.show

def plot_eqoi_vs_dof(plot_dict, show=True, json_path = None, benchmark='uniform'):

    if not json_path:
        json_path = os.getcwd()
    
    print(f'Plot will be saved in: {json_path}')

    # Set up figure
    fig, axes = plt.subplots()

    #get best uniform qoi value as benchmark
    qoi_benchmark = plot_dict[benchmark]['qoi_list'][-1]

    # Extract metadata from the first item in plot_dict (assuming all methods have consistent metadata)
    first_method = next(iter(plot_dict.values()))
    qoi_name = first_method['qoi_name']
    qoi_unit = first_method['qoi_unit']
    dof_name = first_method['dof_name']
    dof_unit = first_method['dof_unit']
    
    # Initialize variables for plot limits
    dof_min, dof_max = float('inf'), float('-inf')
    qoi_min, qoi_max = float('inf'), float('-inf')

    # Initialize variables for plot limits
    dof_min, dof_max = _set_min_max()
    qoi_min, qoi_max  = _set_min_max()

    # Plot the data for each method
    for key, value in plot_dict.items():
        dof_min, dof_max = _set_min_max([dof_min, dof_max],value['dof_list'])
        qoi_min, qoi_max = _set_min_max([qoi_min, qoi_max],value['qoi_list'])
        _dofs, _qois = joint_sort_lists([value['dof_list'],value['qoi_list']],0)

        y = abs((np.array(_qois)-qoi_benchmark)/qoi_benchmark)
        x= _dofs

        print(key,y,x)
    
        
        axes.scatter(x, y, color = value['color'])
        if len(x)>1:
            a,b = np.polyfit(np.log(x[:-1]), np.log(y[:-1]),1)
            axes.loglog(x,x**a*np.exp(b), '.', label=key, 
                color=value['color'], linestyle=value['linestyle'])
    # visualize slope of convergence
    annotation.slope_marker((3e5,1e-2), (-1.2,1), ax=axes, size_frac=0.25, pad_frac=0.1, 
                    text_kwargs = dict(fontsize=14)
                    )
    
    # Set labels and grid
    axes.set_xlabel("DoF count")
    axes.set_ylabel(f"QoI error ($\%$)")
    axes.grid(True)

    # Adjust layout, add legend, and save the figure
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(json_path, 'qoi_vs_dof.jpg'))

    #Display the plot if requested
    if show:
        plt.show()

def plot_eqoi_vs_cpu(plot_dict, show=True, json_path = None, benchmark='uniform'):

    if not json_path:
        json_path = os.getcwd()
    
    print(f'Plot will be saved in: {json_path}')

    # Set up figure
    fig, axes = plt.subplots()

    #get best uniform qoi value as benchmark
    qoi_benchmark = plot_dict[benchmark]['qoi_list'][-1]

    # Extract metadata from the first item in plot_dict (assuming all methods have consistent metadata)
    first_method = next(iter(plot_dict.values()))
    qoi_name = first_method['qoi_name']
    qoi_unit = first_method['qoi_unit']

    # Initialize variables for plot limits
    qoi_min, qoi_max = _set_min_max()
    cpu_min, cpu_max  = _set_min_max()

    # Plot the data for each method
    for key, value in plot_dict.items():

        _qois, _cpus = joint_sort_lists([value['qoi_list'],value['ftime_list']],0)

        y = abs((np.array(_qois)-qoi_benchmark)/qoi_benchmark)
        x= _cpus
        
        qoi_min, qoi_max = _set_min_max([qoi_min, qoi_max],y)
        cpu_min, cpu_max = _set_min_max([cpu_min, cpu_max],x)
        print(key,y,x)
    
        
        axes.scatter(x, y, color = value['color'], label=key)
        if key == 'uniform':
            a,b = np.polyfit(np.log(x[:1]), np.log(y[:1]),1)
            axes.loglog(x,x**a*np.exp(b), '.', label=key, 
                color=value['color'], linestyle=value['linestyle'])
    
    # Set labels and grid
    axes.set_xlabel(f"CPU time ($\mathrm{{s}}$)")
    axes.set_ylabel(f"QoI error ($\%$)")
    # axes.set_ylabel(f"{qoi_name} ($\mathrm{{{qoi_unit}}}$)")
    # axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
    axes.grid(True)

    # Adjust layout, add legend, and save the figure
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(json_path, 'qoi_vs_dof.jpg'))

    #Display the plot if requested
    if show:
        plt.show

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
            for root, _ , files in os.walk(rootpath) 
            for file in files 
            if file.endswith(end_pattern)]

def joint_sort_lists(list_of_lists, index_sortlist=0):
    """
    sort multiple lists based on the ordering of one of them set by the index
    """
    return [list(l) for l in zip(*sorted(zip(*list_of_lists), key=lambda x: x[index_sortlist]))]


if __name__ == "__main__":
    # test script
    json_path = r"/data0/nn_adapt/output/Dec2024_gnn_test_fix_boundary/target_4000/offset/base/"
    
    files = create_logfile_list(json_path, "timesout.txt")
    for file in files:
        export_plot_params_json(file, json_path)

    plot_dict = create_plot_dictionary(json_path)

    plot_qoi_vs_dof(plot_dict, json_path=json_path)
    plot_cpu_vs_dof(plot_dict, json_path=json_path)
    plot_eqoi_vs_dof(plot_dict, json_path=json_path)
    plot_eqoi_vs_cpu(plot_dict, json_path=json_path)