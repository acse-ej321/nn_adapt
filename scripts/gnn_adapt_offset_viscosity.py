import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq
# 

target_complexity = 4000 #TEST

## UNIFORM --------------------------------------------------------------
# rootfolder = "/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/uniform"
# input_params = {
#     "coordinates": [(456, 232), (744, 268)],
#                 "inflow_speed":-5,
#                 }
# sim_uniform = Simulation(Model,rootfolder, parameters = input_params)

# sim_uniform.params["adaptor_method"] = 'uniform'
# sim_uniform.params["indicator_method"] = "none"
# print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')


# sim_uniform.params['miniter'] = 2
# sim_uniform.params['maxiter'] = 3


# logging.info(sim_uniform.params)

# sim_uniform.run_fp_loop() 

# # BASE - GO ANISOTROPIC --------------------------------------------------------------
for vis in [10,]:
    rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/viscosity_{str(vis)}/goal_based"
    input_params = {
        "coordinates": [(456, 232), (744, 268)],
        "viscosity_coefficient": vis,
        }
    sim_uniform = Simulation(Model,rootfolder, parameters = input_params)
    sim_uniform.params["adaptor_method"] = 'steady_anisotropic'
    sim_uniform.params["indicator_method"] = "none"
    print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    sim_uniform.params['miniter'] = 3
    sim_uniform.params['maxiter'] = 14
    sim_uniform.params["base"]= 400 # changed from 200
    sim_uniform.params["target"]= target_complexity

    logging.info(sim_uniform.params)
    sim_uniform.run_fp_loop() 

# ## MLP - GO ANISOTROPIC -  - ------------------------------------------------------------
#     rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/viscosity_{str(vis)}/ml_mlp"
#     input_params = {
#         "coordinates": [(456, 232), (744, 268)],
#         "viscosity_coefficient": vis,
#         }
#     sim_uniform = Simulation(Model,rootfolder, parameters = input_params)
#     sim_uniform.params["adaptor_method"] = 'steady_anisotropic'
#     sim_uniform.params["indicator_method"] = "mlp"
#     print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

#     sim_uniform.params['miniter'] = 3
#     sim_uniform.params['maxiter'] = 14
#     sim_uniform.params["base"]= 400 # changed from 200
#     sim_uniform.params["target"]= target_complexity

#     logging.info(sim_uniform.params)
#     sim_uniform.run_fp_loop() 


# ## GNN ALL- GO ANISOTROPIC -  - ------------------------------------------------------------
#     rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/viscosity_{str(vis)}/ml_gnn"
#     input_params = {
#         "coordinates": [(456, 232), (744, 268)],
#         "viscosity_coefficient": vis,
#         }
#     sim_uniform = Simulation(Model,rootfolder, parameters = input_params)
#     sim_uniform.params["adaptor_method"] = 'steady_anisotropic'
#     sim_uniform.params["indicator_method"] = "gnn"
#     print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

#     sim_uniform.params['miniter'] = 3
#     sim_uniform.params['maxiter'] = 14
#     sim_uniform.params["base"]= 400 # changed from 200
#     sim_uniform.params["target"]= target_complexity

#     logging.info(sim_uniform.params)
#     sim_uniform.run_fp_loop() 


