import sys
sys.path.append("..")
from workflow import *
from models import TurbineMeshSeq

Model = TurbineMeshSeq
# 

base_complexity = 400
target_complexity = 4000 #TEST
for speed in [5,-5]:
    input_params = {
        "coordinates": [(456, 268), (744,232)],
                "inflow_speed":speed,
                }
## UNIFORM --------------------------------------------------------------
    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/uniform"
    # rootfolder = f"/home/phd01/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/uniform"

    # sim_uniform = Simulation(Model,rootfolder, parameters = input_params)

    # sim_uniform.params["adaptor_method"] = 'uniform'
    # sim_uniform.params["indicator_method"] = "none"
    # print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')


    # sim_uniform.params['miniter'] = 2
    # sim_uniform.params['maxiter'] = 3


    # logging.info(sim_uniform.params)
    # sim_uniform.run_simulation()
    # sim_uniform.run_fp_loop() 

# # BASE - HESSIAN --------------------------------------------------------------
    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/hessian"
    # rootfolder = f"/home/phd01/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/hessian"

    # sim_uniform = Simulation(Model,rootfolder, parameters = input_params)
    # sim_uniform.params["adaptor_method"] = 'hessian'
    # sim_uniform.params["indicator_method"] = "none"
    # print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    # sim_uniform.params['miniter'] = 3
    # sim_uniform.params['maxiter'] = 14
    # sim_uniform.params["base"]= base_complexity # changed from 200
    # sim_uniform.params["target"]= target_complexity

    # logging.info(sim_uniform.params)
    # sim_uniform.run_simulation()
    # sim_uniform.run_fp_loop() 


# # BASE - GO ISOTROPIC --------------------------------------------------------------

    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/viscosity_{str(vis)}/goal_based"
    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/goal_based"
    # rootfolder = f"/home/phd01/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/goal_based"

    # sim_uniform = Simulation(Model, rootfolder, parameters = input_params)
    # sim_uniform.params["adaptor_method"] = 'isotropic'
    # sim_uniform.params["indicator_method"] = "none"
    # print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    # sim_uniform.params['miniter'] = 3
    # sim_uniform.params['maxiter'] = 14
    # sim_uniform.params["base"]= base_complexity # changed from 200
    # sim_uniform.params["target"]= target_complexity

    # logging.info(sim_uniform.params)
    # sim_uniform.run_simulation()
    # sim_uniform.run_fp_loop() 

# # BASE - GO ANISOTROPIC --------------------------------------------------------------

    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/viscosity_{str(vis)}/goal_based"
    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/goal_based"
    rootfolder = f"/home/phd01/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/goal_based"

    sim_uniform = Simulation(Model, rootfolder, parameters = input_params)
    sim_uniform.params["adaptor_method"] = 'anisotropic'
    sim_uniform.params["indicator_method"] = "none"
    print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    sim_uniform.params['miniter'] = 3
    sim_uniform.params['maxiter'] = 14
    sim_uniform.params["base"]= base_complexity # changed from 200
    sim_uniform.params["target"]= target_complexity

    logging.info(sim_uniform.params)
    sim_uniform.run_simulation()
    # sim_uniform.run_fp_loop() 

# ## MLP - GO ANISOTROPIC -  - ------------------------------------------------------------
    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/ml_mlp"

    # sim_uniform = Simulation(Model,rootfolder, parameters = input_params)
    # sim_uniform.params["adaptor_method"] = 'steady_anisotropic'
    # sim_uniform.params["indicator_method"] = "mlp"
    # print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    # sim_uniform.params['miniter'] = 3
    # sim_uniform.params['maxiter'] = 14
    # sim_uniform.params["base"]= base_complexity # changed from 200
    # sim_uniform.params["target"]= target_complexity

    # logging.info(sim_uniform.params)
    # sim_uniform.run_fp_loop() 


# ## GNN ALL- GO ANISOTROPIC -  - ------------------------------------------------------------
    # rootfolder = f"/data0/nn_adapt/output/Nov2024_gnn_test/target_4000/offset/reverse/speed_{str(speed)}/ml_gnn"

    # sim_uniform = Simulation(Model,rootfolder, parameters = input_params)
    # sim_uniform.params["adaptor_method"] = 'steady_anisotropic'
    # sim_uniform.params["indicator_method"] = "gnn"
    # print(f'\n\nmethod: {sim_uniform.params["adaptor_method"] }, {sim_uniform.params["indicator_method"]} ')

    # sim_uniform.params['miniter'] = 3
    # sim_uniform.params['maxiter'] = 14
    # sim_uniform.params["base"]= base_complexity# changed from 200
    # sim_uniform.params["target"]= target_complexity

    # logging.info(sim_uniform.params)
    # sim_uniform.run_fp_loop() 


