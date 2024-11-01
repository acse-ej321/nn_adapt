import firedrake as fd
# from firedrake.__future__ import interpolate
import animate as ani 
import goalie as gol 
import goalie_adjoint as gol_adj
from animate.utility import VTKFile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker

# import pt_discharge_features as features

import numpy as np
import pickle # ej321

import os as os
from pathlib import Path
from collections.abc import Callable

import pdb #ej321

import importlib
from time import perf_counter
from matplotlib.pyplot import cm

from pyadjoint import stop_annotating

# timing functions manually:
import logging
from time import perf_counter

from firedrake.pyplot import * # ej321 - talk about namespace issue...

import firedrake.cython.dmcommon as dmcommon

# from workflow.utility import *
from workflow.features import *

class Adaptor:

    def __init__(self, mesh_seq, parameters, filepath= None, *kwargs):
        self.filepath = filepath if filepath else os.getcwd() # can call directly for filepath
        self.mesh_seq = mesh_seq
        self.params = parameters
        print(self.params)
        # self.method = mesh_seq.params["adaptor_method"] if mesh_seq.params["adaptor_method"] is not None else "uniform"
        self.metrics = None 
        self.hessians = None
        # self.indicators = [] #to record metrics - may need to be temporary, here for analysis
        # self.solutions = [] # intermediate record as well? - may get large?? delete as needed, just an idea
        self.complexities = []
        self.target_complexities = []
        self.mesh_stats=[]
        self.qois=[]
        self.f_out=None
        self.a_out=None
        self.i_out=None
        self.m_out=None
        self.h_out=None
        # self.set_outfolder()
        self.adapt_iteration = 0
        self.boundary_labels={}
        self.local_filepath=None
       
        




    def set_outfolder(self, suffix=""):
        # TODO: fix to only create file if actually output

        self.local_filepath = f"{self.filepath}/{self.params['adaptor_method']}_{suffix}"
        if  not os.path.isdir(self.local_filepath):
            os.makedirs(self.local_filepath) 
            

        # print(f"{self.filepath}{self.params}{suffix}")
        # self.f_out= VTKFile(f"{self.filepath}/{self.params['adaptor_method']}_{suffix}/forward.pvd")
        # self.a_out= VTKFile(f"{self.filepath}/{self.params['adaptor_method']}_{suffix}/adjoint.pvd")
        # self.i_out= VTKFile(f"{self.filepath}/{self.params['adaptor_method']}_{suffix}/indicator.pvd")
        # self.m_out= VTKFile(f"{self.filepath}/{self.params['adaptor_method']}_{suffix}/metric.pvd")
        # self.h_out= VTKFile(f"{self.filepath}/{self.params['adaptor_method']}_{suffix}/hessian.pvd")
        
        # temporarily set this as the current working directory
        os.chdir(self.local_filepath)
        


    def update_method(method = None):
        if method is not None:
            self.method = method


    def set_fixed_area(self, mesh):
        """
        Fix an area of the mesh at a cell level from adapting
        """
        # flag to fix the boundary
        fix_area = self.params["fix_area"]
        print(f"set boundary {fix_area}")

        if fix_area:
            # the id of the boundary line - only set up now to fix one id
            area_labels = self.params["area_labels"]

            _index = np.max(mesh.topology_dm.getLabelIdIS(dmcommon.CELL_SETS_LABEL).indices)+10
            # print(f"for boundary fix label id start at {_index}")

            for label_id in area_labels:
                # initialize list to store new boundary segment ids
                self.area_labels[label_id] = []
                # get the edge id's associated witht he boundary
                group_orig=mesh.topology_dm.getStratumIS(
                    dmcommon.CELL_SETS_LABEL,label_id).indices
                
                # check that something to reassign
                if group_orig.size > 0:
                    # # user specified starting index for where to start relabeling
                    # _index = self.params["boundary_start_i"]
                    # loop and add markers for each cell individually in the mesh
                    for el in group_orig:
                        mesh.topology_dm.clearLabelValue(dmcommon.CELL_SETS_LABEL, el, label_id) # is this needed? YES!!
                        mesh.topology_dm.setLabelValue(dmcommon.CELL_SETS_LABEL, el, _index)
                        # print(f"set {el} el to {_index} id")
                        # save new label association with original
                        self.area_labels[label_id].append(_index)
                        _index+=1

                    else:
                        pass
                
                # QC:
                # print(f"face labels after fix : {mesh.topology_dm.getLabelIdIS(dmcommon.CELL_SETS_LABEL).indices}")
                # print(f"face labels after fix : {mesh.topology_dm.getStratumIS(dmcommon.CELL_SETS_LABEL,label_id).indices}")


    def unset_fixed_area(self, mesh):

        # flag to fix the boundary(s)
        fix_area= self.params["fix_area"]
        print(f"unset area {fix_area}")

        

        if fix_area:
            # the id of the boundary line - only set up now to fix one id
            area_label_ids = self.params["area_labels"]

            # get the list of all boundary ids currently in the mesh
            new_labels = list(mesh.topology_dm.getLabelIdIS(dmcommon.CELL_SETS_LABEL).indices)
            dropped = []
            for label_id in area_label_ids:

                for new_id_ in self.area_labels[label_id]:  
                    
                    if new_id_ in new_labels:
                        group_new = mesh.topology_dm.getStratumIS(dmcommon.CELL_SETS_LABEL,new_id_).indices
                        for el_ in group_new:
                            mesh.topology_dm.clearLabelValue(dmcommon.CELL_SETS_LABEL,el_, new_id_) # is this needed? YES!!
                            mesh.topology_dm.setLabelValue(dmcommon.CELL_SETS_LABEL, el_, label_id)
                    else:
                        dropped.append(new_id_)
            
            # QC:
            # print(f"{dropped} dropped by MMG")


    def set_fixed_boundary(self, mesh):

        # flag to fix the boundary
        fix_boundary = self.params["fix_boundary"]
        print(f"set boundary {fix_boundary}")

        _index = np.max(mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices)+10
        

        if fix_boundary:
            # the id of the boundary line - only set up now to fix one id
            boundary_labels = self.params["boundary_labels"]
            # user specified starting index for where to start relabeling
            # _index = self.params["boundary_start_i"]


            _index = np.max(mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices)+10
            # print(f"for boundary fix label id start at {_index}")

            for label_id in boundary_labels:
                # initialize list to store new boundary segment ids
                self.boundary_labels[label_id] = []
                # get the edge id's associated witht he boundary
                group_orig=mesh.topology_dm.getStratumIS(
                    dmcommon.FACE_SETS_LABEL,label_id).indices
                
                # check that something to reassign
                if group_orig.size > 0:
                    # # user specified starting index for where to start relabeling
                    # _index = self.params["boundary_start_i"]
                    # loop and add markers for each cell individually in the mesh
                    for el in group_orig:
                        mesh.topology_dm.clearLabelValue(dmcommon.FACE_SETS_LABEL,el, label_id) # is this needed? YES!!
                        mesh.topology_dm.setLabelValue(dmcommon.FACE_SETS_LABEL, el, _index)
                        # print(f"set {el} el to {_index} id")
                        # save new label association with original
                        self.boundary_labels[label_id].append(_index)
                        _index+=1

                    else:
                        pass
                
                # QC:
                # print(f"face labels after fix : {mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")
                # print(f"face labels after fix : {mesh.topology_dm.getStratumIS(dmcommon.FACE_SETS_LABEL,label_id).indices}")
            

    def unset_fixed_boundary(self, mesh):

        # flag to fix the boundary(s)
        fix_boundary = self.params["fix_boundary"]
        print(f"unset boundary {fix_boundary}")

        

        if fix_boundary:
            # the id of the boundary line - only set up now to fix one id
            boundary_label_ids = self.params["boundary_labels"]

            # get the list of all boundary ids currently in the mesh
            new_labels = list(mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices)
            dropped = []
            for label_id in boundary_label_ids:

                for new_id_ in self.boundary_labels[label_id]:  
                    
                    if new_id_ in new_labels:
                        group_new = mesh.topology_dm.getStratumIS(dmcommon.FACE_SETS_LABEL,new_id_).indices
                        for el_ in group_new:
                            mesh.topology_dm.clearLabelValue(dmcommon.FACE_SETS_LABEL,el_, new_id_) # is this needed? YES!!
                            mesh.topology_dm.setLabelValue(dmcommon.FACE_SETS_LABEL, el_, label_id)
                    else:
                        dropped.append(new_id_)
            # QC:
            # print(f"{dropped} dropped by MMG")


    def space_time_normalise_old(self, metrics, end_time, timesteps, target, p):
        r"""
        Apply :math:`L^p` normalisation in both space and time.
        :arg metrics: list of :class:`firedrake.function.Function`\s
            corresponding to the metric associated with
            each subinterval
        :arg end_time: end time of simulation
        :arg timesteps: list of timesteps specified in each
            subinterval
        :arg target: target *space-time* metric complexity
        :arg p: normalisation order
        """
    #     metrics: List[Function],
    #     end_time: float,
    #     timesteps: List[float],
    #     target: float,
    #     p: float,
    # ) -> List[Function]:
        # NOTE: Assumes uniform subinterval lengths

        assert p == "inf" or p >= 1.0, f"Norm order {p} not valid"
        num_subintervals = len(metrics)
        assert len(timesteps) == num_subintervals
        dt_per_mesh = [
            fd.Constant(end_time / num_subintervals / dt) for dt in timesteps
        ]

        d = metrics[0].function_space().mesh().topological_dimension()
    
        # Compute global normalisation factor

        integral = 0
        for metric, tau in zip(metrics, dt_per_mesh):
            detM = ufl.det(metric)
            if p == "inf":
                integral += fd.assemble(tau * ufl.sqrt(detM) * ufl.dx)
            else:
                integral += fd.assemble(
                    pow(tau**2 * detM, p / (2 * p + d)) * ufl.dx
                )

        global_norm = fd.Constant(pow(target / integral, 2 / d))
    
        # Normalise on each subinterval

        for metric, tau in zip(metrics, dt_per_mesh):

            determinant = (
                1 if p == "inf" else pow(tau**2 * ufl.det(metric), -1 / (2 * p + d))
            )

            metric.interpolate(global_norm * determinant * metric)

        return metrics


    def dict_mesh_stats(self):
        def_dict = {}
        for m in self.mesh_seq:
            try:
                def_dict['elements'].append(m.num_cells())
                def_dict['vertices'].append(m.num_vertices())
            except:
                def_dict['elements']=[m.num_cells()]
                def_dict['vertices']=[m.num_vertices()]
            for n in ani.quality.QualityMeasure(m)._measures:
                try:
                    measure=ani.quality.QualityMeasure(m)(str(n)).vector().gather()
                    try:
                        def_dict[str(n)].append([measure.mean(),measure.min(),measure.max()])
                    except:
                        def_dict[str(n)]=[[measure.mean(),measure.min(),measure.max()]]
                except:
                    # TO QC:
                    # print(f'error for {str(n)}')
                    pass             
        return def_dict


    def plot_mesh_convergence_stats(self, plot_len=10, subplot_ht=2, show=False):
        
        # these are repeated so just grabbing the first one
        def_dict = self.mesh_stats[0]

        # use the length of the first one to create plots for each
        fig,ax = plt.subplots(len(def_dict)+2, 1,
                figsize=(plot_len,subplot_ht*len(def_dict)),
                sharex=True, layout="constrained")

        print(f'filepath : {self.filepath}')
        # iterate over each statitic category and get key name
        ax[0].plot(self.mesh_seq.estimator_values)
        ax[0].set_ylabel("estimator value")
        qoi_vals = max(self.qois,self.mesh_seq.qoi_values)
        ax[1].plot(qoi_vals)
        ax[1].set_ylabel("qoi value")
        plt.xlabel("iterations")
        for d,key in enumerate(self.mesh_stats[0].keys()): # d in mesh statistic
            
            # get the number of meshes - reused so extract to variable once
            nmesh=len(self.mesh_seq.meshes)
            # create empty lists to hold values for plotting
            x=[]
            y=[]
            y_min =[]
            y_max =[]
            # iterate through the mesh adaption iterations and get dictionary for each
            for index,def_dict in enumerate(self.mesh_stats): # index in interation
                # get the x coordinate - the iteration number
                x.append(index)
                # for each mesh in the mesh sequence             
                for i in range(len(self.mesh_seq.meshes)): # i is mesh number
                    # if the area is a list of values - like vertices or elements
                    if (isinstance(def_dict[key][0], int)):
                        # for the first iteration, build the lists
                        if index ==0:
                            y.append([def_dict[key][i]])
                        # for all other iterations, append to lists
                        else:
                            y[i].append(def_dict[key][i])
                    # if the area is a list of lists
                    else:
                        # for the first iteration, build the lists
                        if index ==0:
                            pass

                            y.append([def_dict[key][i][0]])
                            y_min.append([def_dict[key][i][1]])
                            y_max.append([def_dict[key][i][2]])
                        # for all other iterations, append to lists
                        else:

                            y[i].append(def_dict[key][i][0])
                            y_min[i].append(def_dict[key][i][1])
                            y_max[i].append(def_dict[key][i][2])

            # create a color spectrum to iterate through
            color = iter(cm.gist_rainbow(np.linspace(0, 1, len(self.mesh_seq.meshes))))
            # for each y grouping per mesh, plot the values
            for ny_,y_ in enumerate(y):
                c = next(color)
                ax[d+2].plot(x[:],y_[:], c=c, label=f'mesh {ny_}')
                ax[d+2].set_ylabel(key)
                ax[d+2].legend()

        # assumes the local filepath has been set as the cwd
        local_filepath = os.getcwd()
        print(f'\n\n local filepath {local_filepath}')
        _filepath = os.path.join(
                local_filepath,
                f'mesh_statistics_{self.params["adaptor_method"] }.jpg'
                )
        print(f"fig savign to: {_filepath}")
        fig.savefig(_filepath)

        if show:
            plt.show()


    def plot_adapt_meshes(self,plot_len=16, plot_ht=4, show=False):
        # plt.rcParams['figure.dpi'] = 500
        label='forward'
        field = self.params["field"]

        for i, _sol in enumerate(self.solutions):
            # print(f"PLOT SOL: {_sol[label]}")
            # _sol_obj=_sol
            _sol_obj = _sol.solutions[field][label]
            if np.shape(_sol[field][label])[0]>1:
                rows=np.shape(_sol_obj)[1] # time steps per mesh
                cols=np.shape(_sol_obj)[0] # number of meshes
                # print(rows,cols)
                # just need the first column as all time steps have same mesh
                fig, axes = plt.subplots(rows,cols,figsize=(plot_len, plot_ht))
                for n in range(rows): # just in case change - so don't lose logic, not expensive
                    for m in range(cols):
                        _mesh= _sol_obj[m][n].function_space().mesh()
                        # print(_sol[field][label][m][n],_sol[field][label][m][n].subfunctions)
                        triplot(_mesh, axes=axes[n][m])
                        try:
                            tripcolor(_sol_obj[m][n], axes=axes[n][m], alpha=0.5, edgecolors='r', linewidth=0.0)
                        except:
                            # print("issue with solution field plotting")
                            # continue
                            if len(_sol_obj[m][n].subfunctions)>1:
                                tripcolor(_sol_obj[m][n].subfunctions[1], axes=axes[n][m], alpha=0.5, edgecolors='r', linewidth=0.0)
                            # plt.show()
                        axes[n][m].set_aspect('equal')
                        axes[n][m].set_title(f"Mesh {m}/{i}, {_mesh.num_vertices()} vertices, {_mesh.num_cells()} cells",loc='center', wrap=True)
            
            else:
                # print(f'inner sol {_sol[field][label][-1][-1]}')
                meshes = _sol_obj[-1][-1].function_space().mesh()
                fig, axes = plt.subplots(len(_sol.solutions),1,figsize=(plot_len/2, plot_ht), tight_layout=True)
                triplot(meshes, axes=axes)
                # overlay last solution
                try:
                    tripcolor(_sol_obj[-1][-1], axes=axes, alpha=0.5, edgecolors='r', linewidth=0.0)
                except:
                    print("issue with solution field plotting")
                    # continue
                    if len(_sol_obj[-1][-1].subfunctions)>1:
                        tripcolor(_sol_obj[-1][-1].subfunctions[1], axes=axes, alpha=0.5, edgecolors='r', linewidth=0.0)

                axes.set_title(f"Mesh {i}, {meshes.num_vertices()} vertices, {meshes.num_cells()} cells", loc='center', wrap=True)
            
            # assumes the local filepath has been set as the cwd
            local_filepath = os.getcwd()

            fig.savefig(os.path.join(local_filepath,
                                     f'mesh_{i}of{len(self.solutions)}_{field}_{label}_{self.params["adaptor_method"]}_{self.params["qoi_type"]}.jpg'))
            # print("fig saved")
            if show:
                plt.show()


    def export_pvds_by_mesh(self,field=None):
        # loops ignore timesteps per interval - assumes each interval has just one associated mesh

        if field is None:
            field = self.params["field"]

        # loop through each mesh
        for _m in range(np.shape(self.solutions[0].solutions[field]['forward'])[0]): # of meshes
            fm = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_mesh_{_m}_fwd.pvd'))
            if 'adjoint' in self.solutions[0].solutions[field]:
                am = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_mesh_{_m}_adj.pvd'))
            if not all([a==None for a in self.indicators]):
                im = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_mesh_{_m}_ind.pvd'))
            if len(self.metrics)>0:
                mm = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_mesh_{_m}_met.pvd'))
            for _a in range(np.shape(self.solutions)[0]): # of adaptation steps
                for _t in range(np.shape(self.solutions[_a].solutions[field]['forward'][_m])[0]):
                    try:
                    
                        fm.write(self.solutions[_a].solutions[field]['forward'][_m][_t])
                    except:
                        fm.write(self.solutions[_a].solutions[field]['forward'][_m][_t].subfunctions[0])
                if 'adjoint' in self.solutions[0].solutions[field]:
                    for _t in range(np.shape(self.solutions[_a].solutions[field]['adjoint'][_m])[0]):
                        try:
                            am.write(self.solutions[_a].solutions[field]['adjoint'][_m][_t])
                        except:
                            am.write(*self.solutions[_a].solutions[field]['adjoint'][_m][_t].subfunctions)
                if _a<np.shape(self.solutions)[0]-1:
                    if not all([a==None for a in self.indicators]):
                        im.write(self.indicators[_a].indicators[field][0][0])
                    if len(self.metrics)>0:
                        print(f'length of metrics {len(self.metrics)}, {self.metrics}, {_a}') # ej321 - check
                        mm.write(*self.metrics[_a].subfunctions) # ej321 comment out till fix


    def export_pvds_by_adapt_step(self,field=None):
        # loops ignore timesteps per interval - assumes each interval has just one associated mesh
        
        if field is None:
            field = self.params["field"]

        #loop through each adaptation step
        if np.shape(self.solutions)[0]>1:
            for _a in range(np.shape(self.solutions)[0]): # of adaptation steps
                print(_a)
                fa = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_adapt_{_a}_fwd.pvd'))
                if 'adjoint' in self.solutions[0].solutions[field]:
                    aa = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_adapt_{_a}_adj.pvd'))
                if _a<np.shape(self.solutions)[0]-1:
                    if not all([a==None for a in self.indicators]):
                        ia = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_adapt_{_a}_ind.pvd'))
                    if len(self.metrics)>0:
                        ma = gol.utility.VTKFile(os.path.join(self.filepath,f'_{field}_{self.params["adaptor_method"]}_adapt_{_a}_met.pvd'))
                # ha = gol.utility.VTKFile(f'_ptdis_adapt_{_a}_hes.pvd')
                for _m in range(np.shape(self.solutions[0].solutions[field]['forward'])[0]): # of meshes
                    print(_m)

                    for _t in range(np.shape(self.solutions[_a].solutions[field]['forward'][_m])[0]):
                        try:
                            fa.write(self.solutions[_a][field]['forward'][_m][_t])
                        except:
                            fa.write(self.solutions[_a][field]['forward'][_m][0].subfunctions[1])
                    if 'adjoint' in self.solutions[0][field]:
                        for _t in range(np.shape(self.solutions[_a].solutions[field]['adjoint'][_m])[0]):
                            try:
                                aa.write(self.solutions[_a].solutions[field]['adjoint'][_m][_t])
                            except:
                                aa.write(self.solutions[_a].solutions[field]['adjoint'][_m][_t].subfunctions[1])
                    if _a<np.shape(self.solutions)[0]-1:
                        if not all([a==None for a in self.indicators]):
                            ia.write(self.indicators[_a].indicators[field][_m][0])
                        if len(self.metrics)>0:
                            print(f'length of metrics {len(self.metrics)}') # ej321 - check
                            # ma.write(self.metrics[_a][_m]) # ej321 comment out till fix


    def export_features(self,field=None, ):
            # loops ignore timesteps per interval - assumes each interval has just one associated mesh

            if field is None:
                field = self.params["field"]

            # print(f"TEST {self.metrics}")
            
            # loop through each mesh
            # for _m in range(np.shape(self.solutions[0].solutions[field]['forward'])[0]): # of meshes
            for _m in range(np.shape(self.mesh_seq.solutions[field]['forward'])[0]): # of meshes
                
                # TODO - set as min for number of exports wanted
                # for _a in range(np.shape(self.solutions)[0]): # of adaptation steps
                for _a in range(np.shape(self.mesh_seq.solutions[field]['forward'])[2]):

                    # general statitics per mesh
                    gen_stats={}
                    gen_stats["params"]=self.params
                    gen_stats["mesh_id"] = _m
                    gen_stats["fp_iteration"] = _a
                    # gen_stats["dof"]=sum(np.array([self.solutions[_a].solutions[field]['forward'][_m][0].function_space().dof_count]).flatten())
                    gen_stats["dof"]=sum(np.array([self.mesh_seq.solutions[field]['forward'][_m][0].function_space().dof_count]).flatten())
                    # gen_stats["qoi_value"] = self.mesh_seq.qoi_values[_a]
                    # print(f'\tqois  {self.mesh_seq.qoi_values}\n\t estimators  {self.mesh_seq.estimator_values}')

                    # the forward solution for a mesh
                    fwd_sol = self.solutions[_a][field]['forward'][_m][0]

                    # extract some derivative parameters from the mesh assocated with the forward solution
                    try:
                        d, h1, h2, bnd = extract_mesh_features(fwd_sol)

                        features = {
                            "mesh_d": d,
                            "mesh_h1": h1,
                            "mesh_h2": h2,
                            "mesh_bnd": bnd,
                            "cell_info": get_mesh_info(fwd_sol), # ej321 added
                        }
                    except:
                        print(f'issue with mesh stats exporting features')
                        features = {}

                    # # addtitional parameters for E2N paper
                    # features["global_inflow_speed"]= self.params["inflow_speed"]
                    # features["physics_drag"] =  extract_array(config.parameters.drag(mesh)),
                    # features["physics_viscosity"]= extract_array(config.parameters.viscosity(mesh), project=True),
                    # features["physics_bathymetry"]= extract_array(config.parameters.bathymetry(mesh), project=True),



                    # by timestep
                    # forward solutions degrees of freedom for each time step
                    features["forward_dofs"]={}
                    for _t in range(np.shape(self.solutions[_a].solutions[field]['forward'][_m])[0]):
                            # print(_a, self.mesh_seq.time_partition.end_time,self.mesh_seq.time_partition.num_subintervals)
                            # print(_t, self.mesh_seq.time_partition.num_timesteps_per_export[0],self.mesh_seq.time_partition.timesteps[0])
                            it_sol = _t * self.mesh_seq.time_partition.num_timesteps_per_export[0] * self.mesh_seq.time_partition.timesteps[0]
                            # print(it_sol)
                            
                            features["forward_dofs"][it_sol] = extract_array(self.solutions[_a].solutions[field]['forward'][_m][_t],centroid=True) 
                             

                    # adjoint solution
                    # if adjoint run - extract adjoint and estimator
                    if 'adjoint' in self.solutions[0][field]:
                        try:
                            adj_sol = self.solutions[_a].solutions[field]['adjoint'][_m][0]
                            features["adjoint_dofs"] = extract_array(adj_sol, centroid=True)
                            features["estimator_coarse"] = extract_coarse_dwr_features(self.mesh_seq, fwd_sol, adj_sol, index=0)
                        except:
                            print(f'issue with extracting adjoint')

                    # if not the final solution - for which there sill not be an error indicator
                    # if _a<np.shape(self.solutions)[0]-1:

                        
                    # gen_stats["estimator"]=self.mesh_seq.estimator_values[_a]

                   
                    # add mesh stats into the general stats dictionary 
                    for _key,_value in self.mesh_stats[_a].items():
                        gen_stats[_key]=_value[_m]

                    # if the indicator exists
                    if not all([a==None for a in self.indicators]):
                        indi = self.indicators[_a].indicators[field][0][0]
                        features["estimator"] = indi.dat.data.flatten()
                    
                    # if metrics were captured
                    if len(self.metrics)>0:
                        print(self.metrics[_a])
                        # met = self.metrics[0][_a]
                        # features["metric_dofs"] = get_tensor_at_centroids(met) # ej321 not working, 12 dof intead of 3?

                        # if hessians were captured
                        # if len(self.hessians)>0:
                            # print(f'HESS: {len(self.hessians), len(self.hessians[_a])}')
                            # hess = self.hessians[_a][0]
                            # features["hessian_dofs"]= get_tensor_at_centroids(hess) # ej321 not working, 12 dof intead of 3?


                    # add the general stats to the features
                    features["gen_stats"]=gen_stats

                    # assumes the local filepath has been set as the cwd
                    local_filepath = os.getcwd()

                    # export dictionary to a pickle file
                    output_file =  os.path.join(local_filepath,
                        f'/_{field}_{self.params["adaptor_method"]}_mesh_{_m}_{_a}.pkl'
                            )
                    with open(
                       output_file,'wb') as pickle_file:
                            pickle.dump(features, pickle_file)
                    return output_file


    def get_features(self, field = None):
        '''
        Extract features ala E2N paper along with additional mesh statistics for a 
        SolutionFunction object type which hold just the last solution and related adjoint
        '''

        pass
    

    def export_features_one_iter(self,field=None, ):
        '''
        Export features ala E2N paper along with additional mesh statistics for a 
        SolutionFunction object type which hold just the last solution and related adjoint
        '''

        if field is None:
            field = self.params["field"]
        
        # print(f"TEST {self.metrics}")
        _a = self.mesh_seq.fp_iteration
        # loop through each mesh
        for _m in range(np.shape(self.mesh_seq.solutions[field]['forward'])[0]): # of meshes
            
            # TODO - set as min for number of exports wanted
            # general statitics per mesh
            gen_stats={}
            gen_stats["params"]=self.params
            gen_stats["mesh_id"] = _m
            gen_stats["fp_iteration"] = _a
            print(f'EJ321 - fp iteration in export features {_a}')
            # gen_stats["dof"]=sum(np.array([self.solutions[_a].solutions[field]['forward'][_m][0].function_space().dof_count]).flatten())
            gen_stats["dof"]=sum(np.array([
                self.mesh_seq.solutions[field]['forward'][_m][0].function_space().dof_count
                ]).flatten())
            # gen_stats["qoi_value"] = self.mesh_seq.qoi_values[_a]
            # print(f'\tqois  {self.mesh_seq.qoi_values}\n\t estimators  {self.mesh_seq.estimator_values}')

            # the forward solution for a mesh
            fwd_sol = self.mesh_seq.solutions[field]['forward'][_m][0]

            # extract some derivative parameters from the mesh assocated with the forward solution
            try:
                d, h1, h2, bnd = extract_mesh_features(fwd_sol)

                features = {
                    "mesh_d": d,
                    "mesh_h1": h1,
                    "mesh_h2": h2,
                    "mesh_bnd": bnd,
                    "cell_info": get_mesh_info(fwd_sol), # ej321 added
                }
            except:
                print(f'issue with mesh stats exporting features')
                features = {}


            # print(f"\n\n FEATURE OUTPUT {self.mesh_seq.model_features.keys()}\n\n")
            try:
                for _key,_value in self.mesh_seq.model_features.items():
                    # print(f'in feature output: {_key,_value}')
                    features[_key] = _value
            except:
                print("Issue with accessing model features on Model object - did you mean to define?")

            # by timestep
            ######
            # forward solutions degrees of freedom for each time step
            features["forward_dofs"]={}
            for _t in range(np.shape(self.mesh_seq.solutions[field]['forward'][_m])[0]):
                    # print(_a, self.mesh_seq.time_partition.end_time,self.mesh_seq.time_partition.num_subintervals)
                    # print(_t, self.mesh_seq.time_partition.num_timesteps_per_export[0],self.mesh_seq.time_partition.timesteps[0])
                    it_sol = _t * self.mesh_seq.time_partition.num_timesteps_per_export[0] * self.mesh_seq.time_partition.timesteps[0]
                    it_sol = str(round(it_sol,2))
                    print(it_sol)
                    
                    features["forward_dofs"][it_sol] = extract_array(self.mesh_seq.solutions[field]['forward'][_m][_t],centroid=True) 
            #####

            # adjoint solution
            # if adjoint run - extract adjoint and estimator
            if 'adjoint' in self.mesh_seq.solutions[field]:
                try:
                    adj_sol = self.mesh_seq.solutions[field]['adjoint'][_m][0]
                    print(f'EJ321 - adj in feature export {adj_sol}')
                    features["adjoint_dofs"] = extract_array(adj_sol, centroid=True)
                    print(f'EJ321 - adj in adj_dofs {features["adjoint_dofs"]}')
                    features["estimator_coarse"] = extract_coarse_dwr_features(self.mesh_seq, fwd_sol, adj_sol, index=0)
                    print(f'EJ321 - adj in estimator_coarse {features["estimator_coarse"]}')
                except:
                    print(f'issue with extracting adjoint')

            # if not the final solution - for which there sill not be an error indicator
            # if _a<np.shape(self.solutions)[0]-1:

                
            # gen_stats["estimator"]=self.mesh_seq.estimator_values[_a]

            
            # add mesh stats into the general stats dictionary
            _mesh_stats = self.dict_mesh_stats() 
            for _key,_value in _mesh_stats.items():
                gen_stats[_key]=_value[_m]


            # if the indicator exists
            if not all([a==None for a in self.mesh_seq.indicators.extract(layout="field")[field]]):
                indi = self.mesh_seq.indicators[field][0][0]
                features["estimator"] = indi.dat.data.flatten()
            
            # if metrics were captured
            if self.metrics is not None:
                print(f' FEATURE OUTPUT METRIC {self.metrics}')
                # met = self.metrics[0][_a]
                # features["metric_dofs"] = get_tensor_at_centroids(met) # ej321 not working, 12 dof intead of 3?

                # if hessians were captured
                # if len(self.hessians)>0:
                    # print(f'HESS: {len(self.hessians), len(self.hessians[_a])}')
                    # hess = self.hessians[_a][0]
                    # features["hessian_dofs"]= get_tensor_at_centroids(hess) # ej321 not working, 12 dof intead of 3?


            # add the general stats to the features
            features["gen_stats"]=gen_stats

            # QC output:
            # print(features)

            # export dictionary to a pickle file

            # assumes the local filepath has been set as the cwd
            local_filepath = os.getcwd()

            output_file = os.path.join(local_filepath,
                f'_{field}_{self.params["adaptor_method"]}_mesh_{_m}_{_a}.pkl'
                    )
            with open(
                output_file,'wb') as pickle_file:
                    pickle.dump(features, pickle_file)
            return features



    def output_vtk_for(self, mesh_seq, field = None, ):

        vtk_folder = f"{self.local_filepath}/vtk_files_fpi{self.adapt_iteration}"

        self.f_out = None
        if field is None:
            field = self.params["field"]

        _sol_obj =  mesh_seq.solutions.extract(layout='subinterval')

        # if the list evaluates to true
        if _sol_obj:
            # QC
            # print([_sol[field]['forward'] for _sol in _sol_obj])
            for _sol in _sol_obj:
                
                # output forward solutions
                if _sol[field]["forward"]:
                    
                    #QC:
                    # print(_sol[field]["forward"])
                    
                    if self.f_out is None:
                        self.f_out= VTKFile(f"{vtk_folder}/forward.pvd")
                    for _t in range(np.shape(_sol[field]["forward"])[0]):
                        self.f_out.write(*_sol[field]['forward'][_t].subfunctions)
                                


    def output_vtk_met(self, mesh_seq, field = None, ):
        # to write to local folder
        vtk_folder = f"{self.local_filepath}/vtk_files_fpi{self.adapt_iteration}"
        self.m_out= None

        if field is None:
            field = self.params["field"]

        # output the metric functions
        if self.metrics:
            print(f'in vtk output: metric: {self.metrics}')
            if self.m_out is None:
                self.m_out= VTKFile(f"{vtk_folder}/metric.pvd")

            for _met in self.metrics:
                _met.rename(f'metric')
                self.m_out.write(*_met.subfunctions)
                
                # QC:
                # print(self.metrics)

    def output_vtk_hes(self, mesh_seq, field = None, ):
        # to write to local folder
        vtk_folder = f"{self.local_filepath}/vtk_files_fpi{self.adapt_iteration}"
        self.h_out= None

        if field is None:
            field = self.params["field"]

        # output the hessian functions
        if self.hessians:
            if self.h_out is None:
                self.h_out= VTKFile(f"{vtk_folder}/hessian.pvd")
            for _hes in self.hessians:
                _hes.rename(f'hessian')
                self.h_out.write(*_hes.subfunctions)


    def output_vtk_ind(self, mesh_seq, field = None, ):
        
        # to write to local folder
        vtk_folder = f"{self.local_filepath}/vtk_files_fpi{self.adapt_iteration}"
        self.i_out= None
        
        if field is None:
            field = self.params["field"]

        _ind_obj = mesh_seq.indicators.extract(layout='subinterval')

        if _ind_obj:
            # output the indicator function
            for _ind in _ind_obj:
                if _ind[field]:
                    if self.i_out is None:
                        self.i_out= VTKFile(f"{vtk_folder}/indicator.pvd")
                    for _t in range(np.shape(_ind[field])[0]):
                        self.i_out.write(*_ind[field][_t].subfunctions)

    def output_vtk_adj(self, mesh_seq, field = None, ):

        # to write to local folder
        vtk_folder = f"{self.local_filepath}/vtk_files_fpi{self.adapt_iteration}"

        self.a_out= None

        if field is None:
            field = self.params["field"]

        # QC: 
        # print(len(self.mesh_seq.solutions.extract(layout='subinterval')))
            

        _sol_obj =  mesh_seq.solutions.extract(layout='subinterval')

        # if the list evaluates to true
        if _sol_obj:

            # QC
            # print([_sol[field]['forward'] for _sol in _sol_obj])

            for _sol in _sol_obj:
                
                # output adjoint solutions
                if _sol[field]["adjoint"]:

                    #QC:
                    # print(_sol[field]['adjoint'])
                    # print(f"adjoint shape {np.shape(_sol[field]['adjoint'])}")

                    if self.a_out is None:
                        self.a_out= VTKFile(f"{vtk_folder}/adjoint.pvd")
                    for _t in range(np.shape(_sol[field]["adjoint"])[0]):
                        # u.rename(name = 'elev_2d')
                        # uv.rename(name = 'uv_2d')
                        self.a_out.write(*_sol[field]['adjoint'][_t].subfunctions)

                    





    def output_vtk_old(self, mesh_seq, field = None, ):

        if field is None:
            field = self.params["field"]
        # self.set_outfolder(mesh_seq.fp_iteration)
        # print(f'\n\n VTK output: solution by subinterval: {mesh_seq.solutions.extract(layout="subinterval")}')
        
        # write out paraview exports
        for _sol_obj in mesh_seq.solutions.extract(layout='subinterval'):
            print(f'\n\n VTK output time step check: {np.shape(_sol_obj[field]["forward"])}')
            for _t in range(np.shape(_sol_obj[field]["forward"])[0]):
                self.f_out.write(*_sol_obj[field]['forward'][_t].subfunctions)
                self.a_out.write(*_sol_obj[field]['adjoint'][_t].subfunctions)
        if mesh_seq.indicators is not None:
            # print(indicators.extract(layout='subinterval'))
            for _ind_obj in mesh_seq.indicators.extract(layout='subinterval'):
                for _t in range(np.shape(_ind_obj[field])[0]):
                    self.i_out.write(*_ind_obj[field][_t].subfunctions)
        if self.metrics is not None:
            for _met in self.metrics:
                print(self.metrics)
                self.m_out.write(*_met.subfunctions)
        if self.hessians is not None:
            for _hes in self.hessians:
                self.h_out.write(*_hes.subfunctions)


    def output_checkpoint(self, mesh_seq, field = None, ):

        if field is None:
            field = mesh_seq.params["field"]

          # write out Checkpoint
        print(f'FOR CHECKPOINT {mesh_seq.meshes}, {mesh_seq.time_partition.field_types}') 
        for _m, _mesh in enumerate(mesh_seq):
            print(f'\n\n CHECKPOINT: {_m} {_mesh} {_mesh.name} ')
            chkpt_file = os.path.join(os.getcwd(),
                    f'_{field}_{mesh_seq.params["adaptor_method"]}_mesh_{_m}_{mesh_seq.fp_iteration}.h5'
                        )
            with fd.CheckpointFile(chkpt_file, 'w') as afile:
                afile.save_mesh(_mesh)
                _parameters={
                    "params": mesh_seq.params,
                    "mesh_stats":{} 
                }
                # add mesh stats into the general stats dictionary
                _mesh_stats = self.dict_mesh_stats() 
                for _key,_value in _mesh_stats.items():
                    _parameters["mesh_stats"][_key]=_value[_m]

                afile._write_pickled_dict('/', 'parameters', _parameters)

                # save forward solution
                if 'steady' in mesh_seq.time_partition.field_types:
                    print("HERE - steady save for forward")
                    afile.save_function(
                        mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][0]
                        )
                else:
                    print(f"HERE - UNsteady save for forward, range {np.shape(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'])[0]}")
                    for _t in range(np.shape(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'])[0]):
                        it_sol = _t * mesh_seq.time_partition.num_timesteps_per_export[0] * mesh_seq.time_partition.timesteps[0]

                        # print(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][_t].name())
                        afile.save_function(
                            mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][_t],
                            idx = _t
                        )

                # save adjoint - is this needed?
                if mesh_seq.steady == False:
                    afile.save_function(
                        mesh_seq.solutions.extract(layout='subinterval')[_m][field]['adjoint'][0]
                        )

                # save indicators
                if mesh_seq.indicators is not None:
                    afile.save_function(
                        mesh_seq.indicators.extract(layout='subinterval')[_m][field][0]
                        )


    def adapt_outputs(self, mesh_seq, method):

        # field = mesh_seq.params["field"]

        # write out pickle of export mesh and ML parameters
        features = self.export_features_one_iter()

        # output vtk
        self.output_vtk_for(mesh_seq)
        if method not in ["uniform", "steady_hessian", "time_hessian"] and mesh_seq.steady == False:
            self.output_vtk_adj(mesh_seq)

        # output checkpoint
        self.output_checkpoint(mesh_seq)

        return features
      

    def adaptor(self, mesh_seq, solutions = None, indicators=None):

        self.adapt_iteration +=1
        print(f"\n\n ADATPOR ITERATION: {self.adapt_iteration}")
        # field = mesh_seq.params["field"]
        method = mesh_seq.params["adaptor_method"] 
        
        
        # Extract features and Exports
        features = self.adapt_outputs(mesh_seq, method)


        # ML Surrogate for Indicators
        field = mesh_seq.params["field"]
        indimethod = mesh_seq.params["indicator_method"] 

        print(f"\n\n adaptor using: {method} - {indimethod}")
        
        
        # if indicator method specificed, it is recalculated
        # else no change from base
        if indimethod == "gnn":
            self.mesh_seq.indicators[field][0][0] = gnn_indicator_fit(features, mesh_seq.meshes[-1])
        if indimethod == "gnn_noadj":
            self.mesh_seq.indicators[field][0][0] = gnn_noadj_indicator_fit(features, mesh_seq.meshes[-1])
        if indimethod == "mlp":
            self.mesh_seq.indicators[field][0][0] = mlp_indicator_fit(features, mesh_seq.meshes[-1])

        # QC ML method
        # print(f'ml gnn indicator: {self.mesh_seq.indicators }')

        # get method:
        if method == "uniform":
            return self.uniform_adaptor(mesh_seq)
        if method == "steady_hessian":
            return self.steady_hessian_adaptor(mesh_seq)
        if method == "steady_isotropic":
            return self.steady_isotropic_adaptor(mesh_seq)
        if method == "time_hessian":
            return self.time_hessian_adaptor(mesh_seq)
        if method == "time_isotropic":
            return self.time_isotropic_adaptor(mesh_seq)
        if method == "steady_anisotropic":
            return self.steady_anisotropic_adaptor(mesh_seq)
        if method == "time_anisotropic":
            return self.time_anisotropic_adaptor(mesh_seq)
        
        # default to uniform
        return self.uniform_adaptor(mesh_seq)



    # @timeit
    def uniform_adaptor(self,mesh_seq):
    
        # paramters
        label='forward'
        field = mesh_seq.params["field"]

        # fields
        solutions = mesh_seq.solutions
        
        # FOR ANALYSIS
        #parameters
        iteration = mesh_seq.fp_iteration
        _qoi = fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            

        # ej321 - try skipping for memory - to get more complex meshes
        # if iteration == 0:
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats()) 
        print(f"\n\t current mesh: {self.mesh_stats[-1]} elements")
        # has to be before adaptation so solution and mesh in same Space
        self.qois.append(_qoi)

        print(f'\nQOI: {_qoi}\n')
        logging.info(f'QOI, {_qoi}, uniform')


        # timer start:
        duration = -perf_counter()

        # only run to min number of iterations

        if iteration >= mesh_seq.params["miniter"]:
        # if iteration > mesh_seq.params["miniter"]:
            print(f'\n reached max uniform iterations at {mesh_seq.fp_iteration} base on min iter {mesh_seq.params["miniter"]}')
            return False

        # Adapt 
        for i, mesh in enumerate(mesh_seq):
            
            # use mesh hierarchy to halve mesh spacing until converged
            if not mesh_seq.converged[i]:
                print(f'new mesh name mesh_{i}_{iteration+1}')
                mh = fd.MeshHierarchy(mesh_seq.meshes[i],1)
                mesh_seq.meshes[i] = mh[-1]
                mesh_seq.meshes[i].name = f'mesh_{i}_{iteration+1}'
            else:
                # Stats post adaptation - final??
                pass
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(fd.assemble(mesh_seq.calc_qoi(-1,mesh_seq.solutions[field][label][-1][-1])))
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, uniform')
        print(f' uniform adaptor time taken: {duration:.6f} seconds')

        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (ndofs, nelem) in enumerate(zip(mesh_seq.count_vertices(), mesh_seq.count_elements())):
            gol_adj.pyrint(
                f"  subinterval {i},"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d}, uniform')
            logging.info(f'ELEMENTS, {i}, {nelem:4d}, uniform')
            # logging.info(f'COMPLEX, {i}, {ndofs:4d}, uniform')

        return True
    
    def steady_hessian_adaptor(self, mesh_seq):


        # print(f'HESSIAN SOL AT START: {mesh_seq.solutions}')

        # paramters
        label='forward'
        field = mesh_seq.params["field"]
        iteration = mesh_seq.fp_iteration

        #fields
        solutions = mesh_seq.solutions       
     
        # FOR ANALYSIS
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            )
        
        # QC
        print(f'\nQOI: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, steady, hessian')
        
        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        ramp_test=list(base+(target-np.geomspace(target,base, 
        num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
        mp = {"dm_plex_metric_target_complexity":ramp_test[iteration],
                "dm_plex_metric_p": 1,
            }
        
        self.target_complexities.append(mp["dm_plex_metric_target_complexity"])

        # function parameters
        self.metrics = []
        complexities = []
        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(mesh_seq.solutions[field][label]):

            # QC:
            print(f"\n in steady hessian, solution loop i{i} ")
            # print(f"- sols_step: {sols_step}")

            # list to hold metrics per time step - to be combined later
            # _metrics=[]
            self.hessians=[]
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\n in steady hessian, solution loop j{j} ")
                # print(f" - sol: {sol}")
            
                # Recover the Hessian of the current solution
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                hessian = hessians[0] # ej321 - set the first hessian to the base
                
                hessian.average(*hessians[1:]) # ej321 - all the other hessians
            
                hessian.set_parameters(mp)
            
                # ej321 - steps not in Joe's workflow?
                # for steady state - space normalisation
                hessian.normalise()
                
                # _metric = hessian

                # append the metric for the step in the time partition
                # _metrics.append(_metric)
                # TODO: just output 
                self.hessians.append(hessian)
                

            _hessians = self.hessians

            # constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT HESSIAN STEADY: \n")
            gol_adj.enforce_variable_constraints(_hessians, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                )

            # QC:
            print(f'constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')

            # print(f'combining {len(_metrics)} metrics for timestep')
            metric =_hessians[0] # ej321 - set the first hessian to the base
            metric.average(*_hessians[1:]) # ej321 - all the other hessians     
        
            # metrics.append(metric)
            self.metrics.append(metric)

        # output metrics to vtk
        self.output_vtk_hes(mesh_seq, mesh_seq.params["adaptor_method"])
        self.output_vtk_met(mesh_seq, mesh_seq.params["adaptor_method"])     
        metrics = self.metrics

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                # print(f'new mesh name mesh_{i}_{iteration+1}')
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                # print(f'new mesh: {mesh_seq.meshes[i].name}')
            else:
                # Stats post adaptation - final??
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(fd.assemble(
                    mesh_seq.calc_qoi(
                        mesh_seq.calc_qoi(-1, solutions)
                        # -1,mesh_seq.solutions[field][label][-1][-1]
                        )))
        
        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, steady, hessian')
        print(f' steady, hessian adaptor time taken: {duration:.6f} seconds')

        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        
        # self.metrics.append(metrics)
        # [self.m_out.write(*m.subfunctions) for m in metrics]
        
        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (target, complexity, ndofs, nelem) in enumerate(zip(len(complexities)*[self.target_complexities[-1]], complexities, num_dofs, num_elem)):
            gol_adj.pyrint(
                f"  subinterval {i}, target: {target:4.0f}, complexity: {complexity:4.0f}"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d}, steady, hessian')
            logging.info(f'ELEMENTS, {i}, {nelem:4d}, steady, hessian')
            logging.info(f'TARGET, {i}, {target:4.0f}, steady, hessian')
            logging.info(f'COMPLEX, {i}, {complexity:4.0f}, steady, hessian')

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(self.complexities) < 0.90 * target
        return continue_unconditionally
    
    def time_hessian_adaptor(self, mesh_seq):
        
        # parameters
        label='forward'
        field = mesh_seq.params["field"]
        iteration = mesh_seq.fp_iteration

        # fields
        solutions = mesh_seq.solutions

        # FOR ANALYSIS
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            )

        print(f'\nQOI: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, time, hessian')
        print(f'\n\n QOI (calc): { fd.assemble(mesh_seq.calc_qoi(-1, solutions))}')


        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        ramp_test=list(base+(target-np.geomspace(target,base, 
        num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
        mp = {
            "dm_plex_metric_target_complexity":ramp_test[iteration],
            "dm_plex_metric_p": 1,

            }
        
        self.target_complexities.append(
            mp["dm_plex_metric_target_complexity"]
            )

        # function parameters
        self.metrics = []
        complexities = []

        #QC
        # print(f'\n\n solutions in mesh seq: {solutions[field][label]}')
        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(solutions[field][label]):

            # QC:
            print(f"\n in time hessian, solution loop i{i} ")
            # print(f"- sols_step: {sols_step}")
            
            # time step
            dt = mesh_seq.time_partition.timesteps[i]

            # list to hold metrics per time step - to be combined later
            self.hessians=[]
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\n in time hessian, solution loop j{j} ")
                # print(f" - sol: {sol}")
            
                # Recover the Hessian of the current solution
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                hessian = hessians[0] # ej321 - set the first hessian to the base
                
                hessian.average(*hessians[1:]) # ej321 - all the other hessians
            
                hessian.set_parameters(mp)
            
                # ej321 - steps not in Joe's workflow?
                # for steady state - space normalisation
                hessian.normalise()
                
                # _hessian = hessian

                # append the metric for the step in the time partition
                # TODO: just output 
                self.hessians.append(hessian)

            _hessians = self.hessians

            # constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT HESSIAN TIME: \n")
            gol_adj.enforce_variable_constraints(_hessians, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                )

            # QC:
            print(f'constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')
            # print(f'combining {len(_metrics)} metrics for timestep')
            metric =_hessians[0] # ej321 - set the first hessian to the base
            
            # should be equivalent to time integration
            metric.average(*_hessians[1:], weights=[dt]*len(_hessians)) # ej321 - all the other hessians     
        
            self.metrics.append(metric)
        
        # output metrics to vtk
        self.output_vtk_hes(mesh_seq, mesh_seq.params["adaptor_method"])
        self.output_vtk_met(mesh_seq, mesh_seq.params["adaptor_method"]) 
        metrics = self.metrics

        print(f'before complexities: {[metric.complexity() for metric in metrics]}')
        ##*****************************************************************************************##
        # Apply space time normalisation
        gol_adj.space_time_normalise(metrics, mesh_seq.time_partition, mp)
        # print(mesh_seq.time_partition.timesteps[i],mesh_seq.time_partition.subintervals[i][1] )
        #QC: metrics, end_time, timesteps, target, p)
        # from IPython import embed; embed()
        # metrics = self.space_time_normalise_old(
            # metrics, 0.5, [0.03125,0.03125,0.03125,0.03125],ramp_test[iteration],1
        # )
        print(f'after complexities: {[metric.complexity() for metric in metrics]}')

        

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                
            
            else:
                # Stats post adaptation - final??
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1, solutions)
                        # mesh_seq.calc_qoi(-1,mesh_seq.solutions[field][label][-1][-1])
                        )
                    )

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, time, hessian')
        print(f' time, hessian adaptor time taken: {duration:.6f} seconds')

        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        # self.metrics.append(metrics)
        # [self.m_out.write(*m.subfunctions) for m in metrics]

        
        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (target, complexity, ndofs, nelem) in enumerate(zip(len(complexities)*[self.target_complexities[-1]], complexities, num_dofs, num_elem)):
            gol_adj.pyrint(
                f"  subinterval {i}, target: {target:4.0f}, complexity: {complexity:4.0f}"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d}, time, hessian')
            logging.info(f'ELEMENTS, {i}, {nelem:4d}, time, hessian')
            logging.info(f'TARGET, {i}, {target:4.0f}, time, hessian')
            logging.info(f'COMPLEX, {i}, {complexity:4.0f}, time, hessian')

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(self.complexities) < 0.90 * target
        return continue_unconditionally
    
    def steady_isotropic_adaptor(self, mesh_seq):
        
        # paramters
        label='forward'
        field = mesh_seq.params["field"]
        iteration = mesh_seq.fp_iteration

        # fields
        solutions = mesh_seq.solutions
        indicators = mesh_seq.indicators

        # FOR ANALYSIS
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            )

        print(f'\nQOI: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, steady, isotropic')
        print(f'\n\n QOI (calc): { fd.assemble(mesh_seq.calc_qoi(-1, solutions))}')
        
        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        ramp_test=list(base+(target-np.geomspace(target,base,
        num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
        mp = {
            "dm_plex_metric_target_complexity":ramp_test[iteration],
            "dm_plex_metric_p": 1,
            # "dm_plex_metric_h_min": 1.0e-04,
            # "dm_plex_metric_h_max": 1.0, # essentially sets it to isotropic 
            }
        
        self.target_complexities.append(
            mp["dm_plex_metric_target_complexity"]
            )

        # function parameters
        self.metrics = []
        complexities = []
        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(solutions[field][label]):

            # QC:
            print(f"\n in steady isotropic, solution loop i{i} ")
            # print(f"- sols_step: {sols_step}")

            # list to hold metrics per time step - to be combined later
            _metrics=[]

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):
                
                # QC:
                print(f"\n in steady isotropic, solution loop i{i} ")
                # print(f"- sols_step: {sols_step}")

                # get local indicator 
                indi = indicators.extract(layout="field")[field][i][j] #update indicators
                estimator = indi.vector().gather().sum()
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an isotropic metric from the error indicator field
                _metric.compute_isotropic_metric(error_indicator=indi, interpolant="L2")
                _metric.normalise()
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)

            # constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT ISO STEADY: \n")
            gol_adj.enforce_variable_constraints(_metrics, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                )

            # QC:
            print(f'constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')
                
            metric =_metrics[0] # ej321 - set the first hessian to the base
            metric.average(*_metrics[1:]) # ej321 - all the other hessians     
        
            # metrics.append(metric)
            self.metrics.append(metric)
            
        # output metrics to vtk
        self.output_vtk_met(mesh_seq, mesh_seq.params["adaptor_method"]) 
        metrics = self.metrics

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:

                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')

            else:
                # Stats post adaptation - final??
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                        )
                    )

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, steady, isotropic')
        print(f' steady, isotropic adaptor time taken: {duration:.6f} seconds')
            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        # self.metrics.append(metrics)
        # self.metrics = metrics
        # [self.m_out.write(*m.subfunctions) for m in metrics]
        # self.hessians.append(_hessians)
        # [self.h_out.write(*h.subfunctions) for h in _hessians]
        
        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (complexity, ndofs, nelem) in enumerate(zip(complexities, num_dofs, num_elem)):
            gol_adj.pyrint(
                f"  subinterval {i}, target: {ramp_test[iteration]:4.0f}, complexity: {complexity:4.0f}"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d}, steady, isotropic')
            logging.info(f'ELEMENTS, {i}, {nelem:4d}, steady, isotropic')
            logging.info(f'TARGET, {i}, {target:4.0f}, steady, isotropic')
            logging.info(f'COMPLEX, {i}, {complexity:4.0f}, steady, isotropic')

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(complexities) < 0.90 * target
        return continue_unconditionally

    def time_isotropic_adaptor(self, mesh_seq):
        
        # paramters
        label='forward'
        field = mesh_seq.params["field"]
        iteration = mesh_seq.fp_iteration

        #fields
        solutions = mesh_seq.solutions
        indicators = mesh_seq.indicators

        # FOR ANALYSIS
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            )
        print(f'\nQOI: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, time, isotropic')       
        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        ramp_test=list(base+(target-np.geomspace(target,base,
        num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
        mp = {
            "dm_plex_metric_target_complexity":ramp_test[iteration],
            "dm_plex_metric_p": 1,
            # "dm_plex_metric_h_min": 1.0e-04,
            # "dm_plex_metric_h_max": 1.0, # essentially sets it to isotropic 
            }
        
        self.target_complexities.append(
            mp["dm_plex_metric_target_complexity"]
            )

        # function parameters
        self.metrics = []
        complexities = []
        # hessians = []
        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(solutions[field][label]):
            
            # QC:
            print(f"\n in unsteady isotropic, solution loop i{i} ")
            # print(f"- sols_step: {sols_step}")


            dt = mesh_seq.time_partition.timesteps[i]
            
            # list to hold metrics per time step - to be combined later
            _metrics=[]
            # _hessians=[]

            # DOES IT NEED TO BE SET BEFORE THE METRIC?
            self.set_fixed_boundary(mesh_seq.meshes[i])
            self.set_fixed_area(mesh_seq.meshes[i])

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            metric = ani.metric.RiemannianMetric(P1_ten)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):
                
                # get local indicator 
                indi = indicators.extract(layout="field")[field][i][j] #update indicators
                estimator = indi.vector().gather().sum()
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an isotropic metric from the error indicator field
                _metric.compute_isotropic_metric(error_indicator=indi, interpolant="L2")
                _metric.normalise()
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)
            
            # constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT ISO TIME: \n")
            gol_adj.enforce_variable_constraints(_metrics, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                )

            # QC:
            print(f'constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')

            # print(f'combining {len(_metrics)} metrics for timestep')
            metric =_metrics[0] # ej321 - set the first hessian to the base
            metric.average(*_metrics[1:], weights=[dt]*len(_metrics)) # ej321 - all the other hessians  
  
        
            # metrics.append(metric)
            self.metrics.append(metric)

        # output metrics to vtk
        self.output_vtk_met(mesh_seq, mesh_seq.params["adaptor_method"]) 

        metrics = self.metrics   

        ##*****************************************************************************************##
        # Apply space time normalisation
        gol_adj.space_time_normalise(metrics, mesh_seq.time_partition, mp)

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:

                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                
                #QC - for fixed mesh segments
                print(f"face labels after adapt : {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

                self.unset_fixed_boundary(mesh_seq.meshes[i])
                self.unset_fixed_area(mesh_seq.meshes[i])

                # QC - for fixed mesh segments
                print(f"face labels after adapt uset : {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

            else:
                # Stats post adaptation - final??
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1,solutions)
                        )
                    )

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, time, isotropic')
        print(f' time, isotropic adaptor time taken: {duration:.6f} seconds')
            
            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        # self.metrics.append(metrics)
        # set the metrics list to be current interation only
        # self.metrics = metrics
        # self.hessians = _hessians
        self.complexities.append(complexities)
        # self.hessians.append(_hessians)
        
        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (complexity, ndofs, nelem) in enumerate(zip(complexities, num_dofs, num_elem)):
            gol_adj.pyrint(
                f"  subinterval {i}, target: {ramp_test[iteration]:4.0f}, complexity: {complexity:4.0f}"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d}, time, isotropic')
            logging.info(f'ELEMENTS, {i}, {nelem:4d}, time, isotropic')
            logging.info(f'TARGET, {i}, {target:4.0f}, time, isotropic')
            logging.info(f'COMPLEX, {i}, {complexity:4.0f}, time, isotropic')

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(complexities) < 0.90 * target

        return continue_unconditionally

    def steady_anisotropic_adaptor(self, mesh_seq):
        
        # paramters
        label='forward'
        field = mesh_seq.params["field"]

        #fields
        solutions = mesh_seq.solutions
        indicators = mesh_seq.indicators
        
        # FOR ANALYSIS
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats())
        self.qois.append(
            fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            )

        print(f'\nQOI: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, steady, anisotropic')   

        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base = mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        iteration = mesh_seq.fp_iteration
        # ramp_test=list(base+(target-np.geomspace(target,base, num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
        mp = {
            # "dm_plex_metric_target_complexity":ramp_test[iteration],
            "dm_plex_metric_target_complexity":gol.ramp_complexity(base, target, iteration), #ej321 try linear ramp complexity for convergence plotting
            "dm_plex_metric_p": 1,
            # "dm_plex_metric_h_min": 1.0e-04,
            # "dm_plex_metric_h_max": 1.0, # essentially sets it to isotropic 
            }
        
        self.target_complexities.append(
            mp["dm_plex_metric_target_complexity"]
            )

        # function parameters
        self.metrics = []
        self.hessians = []
        complexities = []

        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(solutions[field][label]):
            
            # gol_adj.pyrint(f"dt as def for steady state{mesh_seq.time_partition.timesteps[i]}")
            
            # list to hold metrics per time step - to be combined later
            _metrics=[]
            _hessians=[]

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):
                
                # get local indicator 
                indi = indicators[field][i][j]
                estimator = indi.vector().gather().sum()

                # Recover the Hessian of the current solution
                hes_duration = -perf_counter()
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                hessian = hessians[0] # ej321 - set the first hessian to the base
                
                # timer end
                hes_duration += perf_counter()
                # log time:
                logging.info(f'TIMING, hessian, {hes_duration:.6f}, steady, anisotropic')
                print(f' steady, anisotropic hessian time taken: {hes_duration:.6f} seconds')
                


                hessian.average(*hessians[1:]) # ej321 - all the other hessians
                
                # append list for analysis
                self.hessians.append(hessian)
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an anisotropic metric from the error indicator field and the Hessian
                # _metric.compute_anisotropic_dwr_metric(indi, hessian, interpolant="L2")
                _metric.compute_anisotropic_dwr_metric(indi, hessian) #ej321 default interpolant is Clement
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)

            #constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT ANISO STEADY: \n")
            gol_adj.enforce_variable_constraints(_metrics, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                  )

            # QC:
            print(f'constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')

            metric =_metrics[0] # ej321 - set the first hessian to the base
            metric.average(*_metrics[1:]) # ej321 - all the other hessians     
        
            self.metrics.append(metric)
            print(f'\n\n metric = {self.metrics}')

        metrics = self.metrics

        # output metrics to vtk
        self.output_vtk_hes(mesh_seq)
        self.output_vtk_met(mesh_seq)
        self.output_vtk_ind(mesh_seq)
        

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
            else:
                # Stats post adaptation - final??
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                        )
                    )

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, steady, anisotropic')
        print(f' steady, anisotropic adaptor time taken: {duration:.6f} seconds')

            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        
        
        self.complexities.append(complexities)
        # self.metrics.append(metrics)
        # self.metrics = metrics
        # self.hessians = _hessians
        # [self.m_out.write(*m.subfunctions) for m in metrics]
        # self.hessians.append(_hessians)
        # [self.h_out.write(*h.subfunctions) for h in _hessians]
        
        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (complexity, ndofs, nelem) in enumerate(zip(complexities, num_dofs, num_elem)):
            gol_adj.pyrint(
                f"  subinterval {i}, target: {mp['dm_plex_metric_target_complexity']:4.0f}, complexity: {complexity:4.0f}"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d},  steady, anisotropic')
            logging.info(f'ELEMENTS, {i}, {nelem:4d},  steady, anisotropic')
            logging.info(f'TARGET, {i}, {target:4.0f},  steady, anisotropic')
            logging.info(f'COMPLEX, {i}, {complexity:4.0f},  steady, anisotropic')

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(complexities) < 0.90 * target
        return continue_unconditionally
    
    def time_anisotropic_adaptor(self, mesh_seq):
        
        # paramters
        label='forward'
        field = mesh_seq.params["field"]
        iteration = mesh_seq.fp_iteration

        # fields
        solutions = mesh_seq.solutions
        indicators = mesh_seq.indicators

        # QC:
        # print(field, label)
        # print(solutions[field][label][-1][-1])
        
        # FOR ANALYSIS
        # self.solutions.append(solutions)
        # self.indicators.append(indicators)
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                # mesh_seq.calc_qoi(-1)
                mesh_seq.calc_qoi(-1, solutions)
                # mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                )
            )

        # QC:
        print(f'\nQOI: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, time, anisotropic') 
        
        #timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]

        ramp_test=list(base+(target-np.geomspace(target,base,
        num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
        mp = {
            "dm_plex_metric_target_complexity":ramp_test[iteration],
            "dm_plex_metric_p": 1,
    #         "dm_plex_metric_h_min": 1.0e-04,
    #         "dm_plex_metric_h_max": 1.0, # essentially sets it to isotropic 
            }
        
        self.target_complexities.append(
            mp["dm_plex_metric_target_complexity"]
            )

        # function parameters
        self.metrics = []
        self.hessians = []
        complexities = []
        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(solutions[field][label]):

            # QC:
            print(f"\n in steady aniso, solution loop i{i} ")
            # print(f" - sol: {sol}")

            dt = mesh_seq.time_partition.timesteps[i]
            
            # gol_adj.pyrint(f"dt as def for steady state{mesh_seq.time_partition.timesteps[i]}")
            
            # list to hold metrics per time step - to be combined later
            _metrics=[]
            _hessians=[]

            # DOES IT NEED TO BE SET BEFORE THE METRIC?
            self.set_fixed_boundary(mesh_seq.meshes[i])
            self.set_fixed_area(mesh_seq.meshes[i])

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            metric = ani.metric.RiemannianMetric(P1_ten)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\n in steady aniso, solution loop j{j} ")
                # print(f" - sol: {sol}")
                
                # get local indicator 
                indi = indicators[field][i][j]
                estimator = indi.vector().gather().sum()


                # Recover the Hessian of the current solution
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                hessian = hessians[0] # ej321 - set the first hessian to the base
                hessian.average(*hessians[1:]) # ej321 - all the other hessians
                hessian.set_parameters(mp)

                self.hessians.append(hessian)
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an anisotropic metric from the error indicator field
                _metric.compute_anisotropic_dwr_metric(indi, hessian, interpolant="L2")
                
                # normalise the local metric
                # TODO: check if this is the right spot for this
                _metric.normalise()
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)
                
            # print(f'combining {len(_metrics)} metrics for timestep')
            metric =_metrics[0] # ej321 - set the first metric to the base
            
            #constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT ANISO TIME: \n")
            gol_adj.enforce_variable_constraints(_metrics, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                  )

            # QC:
            print(f'constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')


            # CONSIDER THIS CORRECT??
            metric.average(*_metrics[1:], weights=[dt]*len(_metrics)) # ej321 - all the other hessians  
  
        
            # metrics.append(metric)
            self.metrics.append(metric)

            # output metrics to vtk
            self.output_vtk_hes(mesh_seq)
            self.output_vtk_met(mesh_seq)
            self.output_vtk_ind(mesh_seq)

            metrics = self.metrics

        ##*****************************************************************************************##
        
        # Apply space time normalisation
        gol_adj.space_time_normalise(metrics, mesh_seq.time_partition, mp)

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                
                # QC:
                print(f"face labels after adapt : {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

                self.unset_fixed_boundary(mesh_seq.meshes[i])
                self.unset_fixed_area(mesh_seq.meshes[i])

                # QC:
                print(f"face labels after adapt uset : {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

            else:
                # Stats post adaptation - final??
                print("FINAL MESH STATS OUT")
                # self.solutions.append(solutions)
                # self.indicators.append(indicators)
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                        )
                    )

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, time, anisotropic')
        print(f' time, anisotropic adaptor time taken: {duration:.6f} seconds')
            
            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        
        self.complexities.append(complexities)
        # self.metrics.append(metrics)
        # self.metrics = metrics
        # self.hessians = _hessians
        # [self.m_out.write(*m.subfunctions) for m in metrics]
        # self.hessians.append(_hessians)
        # [self.h_out.write(*h.subfunctions) for h in _hessians]
        
        # info
        gol_adj.pyrint(f"adapt iteration {iteration + 1}:")
        for i, (complexity, ndofs, nelem) in enumerate(zip(complexities, num_dofs, num_elem)):
            gol_adj.pyrint(
                f"  subinterval {i}, target: {ramp_test[iteration]:4.0f}, complexity: {complexity:4.0f}"
                f", dofs: {ndofs:4d}, elements: {nelem:4d}"
            )
            # log adaptation
            logging.info(f'DOF, {i}, {ndofs:4d},  time, anisotropic')
            logging.info(f'ELEMENTS, {i}, {nelem:4d},  time, anisotropic')
            logging.info(f'TARGET, {i}, {target:4.0f},  time, anisotropic')
            logging.info(f'COMPLEX, {i}, {complexity:4.0f},  time, anisotropic')

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(complexities) < 0.90 * target
        return continue_unconditionally


if __name__ == "__main__":
    solution = "solution"
    indicator = "indicator"
    # adapt_test = Adaptor(solution)
    # print(adapt_test.solution, adapt_test.indicator)
    # final_solution =mesh_seq.__class__.mro()[3].fixed_point_iteration(mesh_seq,adaptor)
    # mesh_seq.solutions.append(final_solution[0])