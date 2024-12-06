import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq

target_complexity = 4000
input_params = {
    "coordinates": [(456, 232), (744,268)],
            "inflow_speed":10,
            "viscosity_coefficient":2.0,
            "bathymetry_model": "trench",
            }
rootfolder = f"/data0/nn_adapt/output/Dec2024_gnn_test/"

## UNIFORM --------------------------------------------------------------

subfolder = rootfolder+f"target_4000/offset/trench/uniform"

sim_uniform = Simulation(Model,subfolder, parameters = input_params)

sim_uniform.params["adaptor_method"] = 'uniform'
sim_uniform.params["indicator_method"] = "none"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')


sim_uniform.params['miniter'] = 2
sim_uniform.params['maxiter'] = 3


logging.info(sim_uniform.params)
sim_uniform.run_simulation()

# # # BASE - HESSIAN --------------------------------------------------------------
subfolder = rootfolder+f"target_4000/offset/trench/hessian"

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
subfolder = rootfolder+f"target_4000/offset/trench/goal_based"

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

subfolder = rootfolder+f"target_4000/offset/trench/goal_based"

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
subfolder = rootfolder+f"target_4000/offset/trench/ml_mlp"

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
subfolder = rootfolder+f"target_4000/offset/trench/ml_gnn"

sim_uniform = Simulation(Model,subfolder, parameters = input_params)
sim_uniform.params["adaptor_method"] = 'anisotropic'
sim_uniform.params["indicator_method"] = "gnn"
print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

sim_uniform.params['miniter'] = 3
sim_uniform.params['maxiter'] = 14
sim_uniform.params["target"]= target_complexity

logging.info(sim_uniform.params)
sim_uniform.run_simulation()
