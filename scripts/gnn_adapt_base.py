import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq

input_params = {
    "coordinates": [(456, 232), (744,268)],
            }

rootfolder = f"/data0/nn_adapt/output/Dec2024_gnn_test/"

## UNIFORM --------------------------------------------------------------

# subfolder = rootfolder+f"target_4000/offset/base/uniform"

# sim_uniform = Simulation(Model,subfolder, parameters = input_params)

# sim_uniform.params["adaptor_method"] = 'uniform'
# sim_uniform.params["indicator_method"] = "none"
# print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')


# sim_uniform.params['miniter'] = 5
# sim_uniform.params['maxiter'] = 6


# logging.info(sim_uniform.params)
# sim_uniform.run_simulation()


# --------------------------------------------------------------

num_refinements = 12 # targets up to 6400
f= 1

for i in range(num_refinements + 1):

    target_complexity = 100.0 * 2 ** (f * i)

    # # # BASE - HESSIAN --------------------------------------------------------------
    subfolder = rootfolder+f"target_4000/offset/base/hessian"

    sim_uniform = Simulation(Model,subfolder, parameters = input_params)
    sim_uniform.params["adaptor_method"] = 'hessian'
    sim_uniform.params["indicator_method"] = "none"
    print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    sim_uniform.params['miniter'] = 3
    sim_uniform.params['maxiter'] = 14
    sim_uniform.params["target"]= target_complexity

    logging.info(sim_uniform.params)
    sim_uniform.run_simulation()


    # # # BASE - GO ISOTROPIC --------------------------------------------------------------
    subfolder = rootfolder+f"target_4000/offset/base/goal_based"

    sim_uniform = Simulation(Model, subfolder, parameters = input_params)
    sim_uniform.params["adaptor_method"] = 'isotropic'
    sim_uniform.params["indicator_method"] = "none"
    print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    sim_uniform.params['miniter'] = 3
    sim_uniform.params['maxiter'] = 14
    sim_uniform.params["target"]= target_complexity

    logging.info(sim_uniform.params)
    sim_uniform.run_simulation()

    # # # BASE - GO ANISOTROPIC --------------------------------------------------------------

    subfolder = rootfolder+f"target_4000/offset/base/goal_based"

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
    subfolder = rootfolder+f"target_4000/offset/base/ml_mlp"

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
