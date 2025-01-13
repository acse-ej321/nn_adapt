import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq

# EDIT THE ROOT DIRECTORY
rootfolder = "/home/phd01/00_models/nn_adapt/output/"
# rootfolder = f"/data0/nn_adapt/output/""

# EDIT THE MAIN SIMULATION FOLDER
subfolder = rootfolder+f"forward_test/"

sim_uniform = Simulation(Model, subfolder)
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] },\
 {sim_uniform.params["indicator_method"]} \n\n')
logging.info(sim_uniform.params)
sim_uniform.run_simulation()