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

