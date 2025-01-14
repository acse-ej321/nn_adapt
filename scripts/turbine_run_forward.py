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

# sim_uniform = Simulation(Model)
sim_uniform = Simulation(Model, subfolder)
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] },\
 {sim_uniform.params["indicator_method"]} \n\n')
logging.info(sim_uniform.params)
sim_uniform.run_simulation()