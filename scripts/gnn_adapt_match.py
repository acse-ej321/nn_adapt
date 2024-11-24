import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq
# 
# rootfolder = "/home/phd01/00_models/Oct2024_gnn_test/"
rootfolder = "/data0/nn_adapt/output/Nov2024_gnn_test/"

num_refinements = 2 # testing
# f=0.25
f=1 # testing
## UNIFORM --------------------------------------------------------------

# rootfolder = "/data0/00_models/adapt_match/uniform/"


sim_uniform = Simulation(Model,rootfolder)

sim_uniform.params["adaptor_method"] = 'uniform'
sim_uniform.params["indicator_method"] = "none"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')


sim_uniform.params['miniter'] = 2
sim_uniform.params['maxiter'] = 3


logging.info(sim_uniform.params)

sim_uniform.run_fp_loop() 

## Run aligned case
## dof to match Joe: 4000 - 80 000

## BASE - GO ANISOTROPIC --------------------------------------------------------------

# rootfolder = "/data0/00_models/adapt_match/base/"

# num_refinements = 35
# f=0.25
for m in ['steady_anisotropic'
        #   , 'steady_isotropic'
        #   , 'steady_hessian'
          ]: # 
    for i in range(num_refinements+1):
        target_complexity = 1000 #TEST
        # target_complexity = 100.0 * (2**(f * i))
        print(f'\n\njoe loop {i, target_complexity}\n\n')

        sim_uniform = Simulation(Model,rootfolder)
        sim_uniform.params["adaptor_method"] = m
        sim_uniform.params["indicator_method"] = "none"
        print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

        sim_uniform.params['miniter'] = 3
        sim_uniform.params['maxiter'] = 14
        sim_uniform.params["base"]= 400 # changed from 200
        sim_uniform.params["target"]= target_complexity

        logging.info(sim_uniform.params)
        sim_uniform.run_fp_loop() 


## GNN ALL- GO ANISOTROPIC -  - ------------------------------------------------------------
# rootfolder = "/data0/00_models/adapt_match/ml_gnn/"
# num_refinements = 35
num_refinements = 2 # testing
# f=0.25
f=1 # testing
for m in ['steady_anisotropic'
        #   , 'steady_isotropic'
        #   , 'steady_hessian'
          ]: # 
    for i in range(num_refinements+1):
        target_complexity = 1000 #TEST
        # target_complexity = 100.0 * (2**(f * i))
        print(f'\n\njoe loop {i, target_complexity}\n\n')

        sim_uniform = Simulation(Model,rootfolder)
        sim_uniform.params["adaptor_method"] = m
        sim_uniform.params["indicator_method"] = "gnn"
        print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

        sim_uniform.params['miniter'] = 3
        sim_uniform.params['maxiter'] = 14
        sim_uniform.params["base"]= 400 # changed from 200
        sim_uniform.params["target"]= target_complexity

        logging.info(sim_uniform.params)
        sim_uniform.run_fp_loop() 


## MLP - GO ANISOTROPIC -  - ------------------------------------------------------------
# rootfolder = "/data0/00_models/adapt_match/ml_mlp/"
# TODO: issue with extract_coarse_dwr
# num_refinements = 15
# num_refinements = 2 # testing
# f=0.5
for m in ['steady_anisotropic'
        #   , 'steady_isotropic'
          ]: # 
    for i in range(num_refinements+1):
        target_complexity = 1000 #TEST
        # target_complexity = 100.0 * (2**(f * i))
        print(f'\n\njoe loop {i, target_complexity}\n\n')

        sim_uniform = Simulation(Model,rootfolder)
        sim_uniform.params["adaptor_method"] = m
        sim_uniform.params["indicator_method"] = "mlp"
        print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

        sim_uniform.params['miniter'] = 3
        sim_uniform.params['maxiter'] = 14
        sim_uniform.params["base"]= 400 # changed from 200
        sim_uniform.params["target"]= target_complexity

        logging.info(sim_uniform.params)
        sim_uniform.run_fp_loop() 


## GNN NO ADJOINT - GO ANISOTROPIC -  - ------------------------------------------------------------
# rootfolder = "/data0/00_models/adapt_match/ml_gnn_noadj/"
# num_refinements = 15
# num_refinements = 2 # testing
# f=0.5
# for m in ['steady_anisotropic', 'steady_isotropic']: # 
#     for i in range(num_refinements+1):
#         target_complexity = 100.0 * (2**(f * i))
#         print(f'\n\njoe loop {i, target_complexity}\n\n')

#         sim_uniform = Simulation(Model,rootfolder)
#         sim_uniform.params["adaptor_method"] = m
#         sim_uniform.params["indicator_method"] = "gnn_noadj"
#         print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

#         sim_uniform.params['miniter'] = 3
#         sim_uniform.params['maxiter'] = 14
#         sim_uniform.params["base"]= 400 # changed from 200
#         sim_uniform.params["target"]= target_complexity

#         logging.info(sim_uniform.params)
#         sim_uniform.run_fp_loop() 