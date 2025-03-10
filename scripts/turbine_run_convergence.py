import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq

# EDIT THE ROOT DIRECTORY
# rootfolder = "/home/phd01/00_models/nn_adapt/output/"
# rootfolder = f"/data0/nn_adapt/output/""
rootfolder = f"/home/ej321/output"

# EDIT THE MAIN SIMULATION FOLDER
subfolder = rootfolder+f"E2N_compare_Dec2025"

num_refinements = 40
# num_refinements = 2 # testing
f=0.5
for i in range(2,num_refinements+1):
    target_complexity = 100.0 * (2**(f * i))
    print(f'\n\n *** convergence loop {i, target_complexity} *** \n\n')
    input_params = {
        "target": target_complexity
        }
    print(f"input parameters: {input_params}")
    sim_uniform = Simulation(Model, parameters = input_params)
    # sim_uniform = Simulation(Model, subfolder, parameters = input_params)
    print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] },\
    {sim_uniform.params["indicator_method"]} \n\n')
    logging.info(sim_uniform.params)
    sim_uniform.run_simulation()