import firedrake as fd
import animate as ani 
import goalie as gol 
import goalie_adjoint as gol_adj
from animate.utility import VTKFile
import matplotlib.pyplot as plt

import numpy as np
import pickle # ej321

import os as os


from time import perf_counter
from matplotlib.pyplot import cm

# timing functions manually:
import logging

# import firedrake.pyplot as fdplt
import firedrake.cython.dmcommon as dmcommon

# from workflow.utility import *
from workflow.features import extract_mesh_features, get_mesh_info, extract_array, get_hessians
from workflow.features import extract_coarse_dwr_features
from workflow.features import gnn_indicator_fit, gnn_noadj_indicator_fit, mlp_indicator_fit


from collections import defaultdict


class Adaptor:

    def __init__(self, mesh_seq, parameters, filepath= None, *kwargs):
        self.filepath = filepath if filepath else os.getcwd() # can call directly for filepath
        self.mesh_seq = mesh_seq
        self.params = parameters
        self.metrics = None 
        self.hessians = None
        self.complexities = []
        self.target_complexities = []
        self.mesh_stats=[]
        self.qois=[]
        self.f_out=None
        self.a_out=None
        self.i_out=None
        self.m_out=None
        self.h_out=None
        self.adapt_iteration = 0
        self.boundary_labels={}
        self.local_filepath=None
       
    def set_outfolder(self, suffix=""):
        # TODO: fix to only create file if actually output

        self.local_filepath = f"{self.filepath}/{self.params['adaptor_method']}_{suffix}"
        if  not os.path.isdir(self.local_filepath):
            os.makedirs(self.local_filepath) 
        
        # temporarily set this as the current working directory
        os.chdir(self.local_filepath)

        # QC:
        print(f'current working directory changed to: {self.local_filepath}')
        
    def update_method(self, method = None):
        if method is not None:
            self.method = method

    def set_fixed_area(self, mesh):
        """
        Fix an area of the mesh at a cell level from adapting
        """
        # flag to fix the boundary
        fix_area = self.params["fix_area"]

        # QC:
        # print(f"set boundary {fix_area}")

        if fix_area:
            # the id of the boundary line - only set up now to fix one id
            area_labels = self.params["area_labels"]

            _index = np.max(mesh.topology_dm.getLabelIdIS(dmcommon.CELL_SETS_LABEL).indices)+10

            # QC:
            # print(f"for boundary fix label id start at {_index}")

            for label_id in area_labels:
                # initialize list to store new boundary segment ids
                self.area_labels[label_id] = []
                # get the edge id's associated witht he boundary
                group_orig=mesh.topology_dm.getStratumIS(
                    dmcommon.CELL_SETS_LABEL,label_id).indices
                
                # check that something to reassign
                if group_orig.size > 0:
                    # loop and add markers for each cell individually in the mesh
                    for el in group_orig:
                        mesh.topology_dm.clearLabelValue(dmcommon.CELL_SETS_LABEL, el, label_id) # is this needed? YES!!
                        mesh.topology_dm.setLabelValue(dmcommon.CELL_SETS_LABEL, el, _index)

                        # QC:
                        # print(f"set {el} el to {_index} id")

                        # save new label association with original
                        self.area_labels[label_id].append(_index)
                        _index+=1
                
                # QC:
                # print(f"face labels after fix : {mesh.topology_dm.getLabelIdIS(dmcommon.CELL_SETS_LABEL).indices}")
                # print(f"face labels after fix : {mesh.topology_dm.getStratumIS(dmcommon.CELL_SETS_LABEL,label_id).indices}")


    def unset_fixed_area(self, mesh):

        # flag to fix the boundary(s)
        fix_area= self.params["fix_area"]

        # QC:
        # print(f"unset area {fix_area}")

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

        # QC: 
        # print(f"set boundary {fix_boundary}")

        # TODO: fix had coding of index shift
        _index = np.max(mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices)+10

        if fix_boundary:
            # the id of the boundary line - only set up now to fix one id
            boundary_labels = self.params["boundary_labels"]

            # _index = np.max(mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices)+10
            # print(f"for boundary fix label id start at {_index}")

            for label_id in boundary_labels:
                # initialize list to store new boundary segment ids
                self.boundary_labels[label_id] = []
                # get the edge id's associated witht he boundary
                group_orig=mesh.topology_dm.getStratumIS(
                    dmcommon.FACE_SETS_LABEL,label_id).indices
                
                # check that something to reassign
                if group_orig.size > 0:
                    # loop and add markers for each cell individually in the mesh
                    for el in group_orig:
                        mesh.topology_dm.clearLabelValue(dmcommon.FACE_SETS_LABEL,el, label_id) # is this needed? YES!!
                        mesh.topology_dm.setLabelValue(dmcommon.FACE_SETS_LABEL, el, _index)

                        # QC:
                        # print(f"set {el} el to {_index} id")

                        # save new label association with original
                        self.boundary_labels[label_id].append(_index)
                        _index+=1
                
                # QC:
                # print(f"face labels after fix : {mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")
                # print(f"face labels after fix : {mesh.topology_dm.getStratumIS(dmcommon.FACE_SETS_LABEL,label_id).indices}")
            

    def unset_fixed_boundary(self, mesh):

        # flag to fix the boundary(s)
        fix_boundary = self.params["fix_boundary"]

        # QC:
        # print(f"unset boundary {fix_boundary}")

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

    def dict_mesh_stats(self):
        """
        Computes and returns a dictionary containing statistics for each mesh in the sequence.

        This method iterates over a sequence of mesh objects and collects statistics including
        the number of elements and vertices, as well as the mean, minimum, and maximum values 
        for each quality metric.

        :returns: summary mesh statictics in a dictionary

        """
        
        def_dict = defaultdict(list)
        for m in self.mesh_seq:
            def_dict['elements'].append(m.num_cells())
            def_dict['vertices'].append(m.num_vertices())

            # added explicitly as cxx seemed to not be filtering 3d facets out of 2d
            qmeasures = (
                "min_angle",
                "area",
                # "volume",
                # "facet_area",
                "aspect_ratio",
                "eskew",
                "skewness",
                "scaled_jacobian",
                # "metric",
            )
            
            for n in qmeasures:
                measure=ani.quality.QualityMeasure(m)(str(n)).vector().gather()
                def_dict[str(n)].append([measure.mean(),measure.min(),measure.max()])
            
            
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
                for i in range(nmesh): # i is mesh number
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
            color = iter(cm.gist_rainbow(np.linspace(0, 1, nmesh)))
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
        print(f"fig saving to: {_filepath}")
        fig.savefig(_filepath)

        if show:
            plt.show()

    def export_features_one_iter(self,field=None, ):
        '''
        Export features ala E2N paper along with additional mesh statistics for a 
        SolutionFunction object type which hold just the last solution and related adjoint
        '''

        if field is None:
            field = self.params["field"]
        
        _a = self.mesh_seq.fp_iteration

        # QC:
        print(f'Exporting features - for fixed point iteration {_a}')

        # loop through each mesh
        for _m in range(np.shape(self.mesh_seq.solutions[field]['forward'])[0]): # of meshes
            
            # TODO - set as min for number of exports wanted
            # general statitics per mesh
            gen_stats={}
            gen_stats["params"]=dict(self.params) # need to convert from AttrDict back to dict
            gen_stats["mesh_id"] = _m
            gen_stats["fp_iteration"] = _a           
            gen_stats["dof"]=sum(np.array([
                self.mesh_seq.solutions[field]['forward'][_m][0].function_space().dof_count
                ]).flatten())

            # the forward solution for a mesh
            fwd_sol = self.mesh_seq.solutions[field]['forward'][_m][0]

            # extract some derivative parameters from the mesh associated with the forward solution
            d, h1, h2, bnd = extract_mesh_features(fwd_sol)
            # QC:
            print('\t exporting features: extracting derivative features')

            features = {
                "mesh_d": d,
                "mesh_h1": h1,
                "mesh_h2": h2,
                "mesh_bnd": bnd,
                "cell_info": get_mesh_info(fwd_sol), # ej321 added
            }
            # QC:
            print('\t exporting features: exporting derivative and cell info features')

            for _key,_value in self.mesh_seq.model_features.items():
                # QC:
                # print(f'\t exporting features: in feature output: {_key,_value}')
                features[_key] = _value
                #  if not print("Issue with accessing model features on Model object - did you mean to define?")

            # forward solutions degrees of freedom for each time step
            features["forward_dofs"]={}
            for _t in range(np.shape(self.mesh_seq.solutions[field]['forward'][_m])[0]):
                    it_sol = _t * self.mesh_seq.time_partition.num_timesteps_per_export[0] * self.mesh_seq.time_partition.timesteps[0]
                    it_sol = str(round(it_sol,2))
                    # QC:
                    # print(_a, self.mesh_seq.time_partition.end_time,self.mesh_seq.time_partition.num_subintervals)
                    # print(_t, self.mesh_seq.time_partition.num_timesteps_per_export[0],self.mesh_seq.time_partition.timesteps[0])
                    # print(it_sol)
                    features["forward_dofs"][it_sol] = extract_array(self.mesh_seq.solutions[field]['forward'][_m][_t],centroid=True) 

            # adjoint solution
            # if adjoint run - extract adjoint and estimator
            if 'adjoint' in self.mesh_seq.solutions[field]:
                adj_sol = self.mesh_seq.solutions[field]['adjoint'][_m][0]
                print(f'\t exporting features: get adjoint solution {adj_sol.dat.data[:]}')

                features["adjoint_dofs"] = extract_array(adj_sol, centroid=True)
                print('\t exporting features: extract adjoint dof')
                features["estimator_coarse"] = extract_coarse_dwr_features(self.mesh_seq, fwd_sol, adj_sol, index=0)
                print('\t exporting features: extract coarse dwr estimator')

            # add mesh stats into the general stats dictionary
            _mesh_stats = self.dict_mesh_stats() 
            print('\t exporting features: add mesh stats')
            for _key,_value in _mesh_stats.items():
                gen_stats[_key]=_value[_m]

            

            # if the indicator exists
            if not all([a is None for a in self.mesh_seq.indicators.extract(layout="field")[field]]):
                indi = self.mesh_seq.indicators[field][0][0]
                features["estimator"] = indi.dat.data.flatten()
                print('\t exporting features: extracting estimator from indicator')
            
            # if metrics were captured
            if self.metrics is not None:
                print('\t exporting features: extracting metrics - NOT IMPLIMENTED YET')
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

            # assumes the local filepath has been set as the cwd
            # TODO: clean up this code
            local_filepath = os.getcwd()

            output_file = os.path.join(local_filepath,
                f'_{field}_{self.params["adaptor_method"]}_mesh_{_m}_{_a}.pkl'
                    )
            with open(
                output_file,'wb') as pickle_file:
                    pickle.dump(features, pickle_file)

            # QC:
            print(f'\t exporting features: features output to {output_file}')

            return features
        
    #TODO: refactor output statments to be more pythonic - 

    # def _output_selection(output="forward", format="vtk", **kwargs):
    #     function_options = {
    #         "forward": ,
    #         "adjoint": ,
    #         "metric": ,
    #         "hessian": ,
    #         "indicator": ,
    #     }
    #     try:
    #         return function_options[output](output, format, **kwargs)
    #     except KeyError as e:
    #         raise ValueError(f"OUtput '{output}' not currently supported.") from e
        
    # def _output_function(self, output, field= None):
    #     """
    #     Output function to vtk, either for steady or unsteady
    #     sets the output file and writes to it
    #     """

    #     # to write to local folder
    #     vtk_folder = f"{self.local_filepath}/vtk_files_fpi{self.adapt_iteration}"

    #     file_out = None
    #     if field is None:
    #         field = self.params["field"]

    #     # output the function
    #     if file_out is None:
    #         file_out= VTKFile(f"{vtk_folder}/{output}.pvd")


    # def _check_output_for(sol_obj):
    #     """
    #     check that forward solution exists in the solution object for all solutions
    #     """

    #     if sol_obj:
    #         for _sol in _sol_obj:
    #             # output forward solutions
    #             if _sol[field]["forward"]:

    # def _output_for(field):
    #     """
    #     gets forward solution and interates for vtk output
    #     """
    #     # get field
    #     _sol_obj =  mesh_seq.solutions.extract(layout='subinterval')

    #     # if the list evaluates to true
    #     if _check_output_for():
    #         for _t in range(np.shape(_sol[field]["forward"])[0]):
    #             return _sol[field]['forward'][_t].subfunctions
    #             # self.f_out.write(*_sol[field]['forward'][_t].subfunctions)


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
                _met.rename('metric')
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
                _hes.rename('hessian')
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
                    print(_sol[field]['adjoint'])
                    print(f"adjoint shape {np.shape(_sol[field]['adjoint'])}")

                    if self.a_out is None:
                        self.a_out= VTKFile(f"{vtk_folder}/adjoint.pvd")
                    for _t in range(np.shape(_sol[field]["adjoint"])[0]):
                        # u.rename(name = 'elev_2d')
                        # uv.rename(name = 'uv_2d')
                        self.a_out.write(*_sol[field]['adjoint'][_t].subfunctions)

    def output_checkpoint(self, mesh_seq, field = None, ):

        if field is None:
            field = mesh_seq.params["field"]

        # write out Checkpoint
        # QC:
        print(f'Writing Checkpoint: {mesh_seq.meshes}, {mesh_seq.time_partition.field_types}') 
        for _m, _mesh in enumerate(mesh_seq):
            # QC:
            print(f'\t checkpoint - saving: {_m} {_mesh} {_mesh.name} ')
            chkpt_file = os.path.join(os.getcwd(),
                    f'_{field}_{mesh_seq.params["adaptor_method"]}_mesh_{_m}_{mesh_seq.fp_iteration}.h5'
                        )
            with fd.CheckpointFile(chkpt_file, 'w') as afile:
                afile.save_mesh(_mesh)
                print(f'output params: {type(dict(mesh_seq.params))}')
                _parameters={
                    "params": dict(mesh_seq.params), # ej321 test
                    "mesh_stats":{} 
                }
                
                # _parameters = mesh_seq.params.to_dict() # convert AttrDict back to dict type
                # _parameters["mesh_stats"]={}
                # add mesh stats into the general stats dictionary
                _mesh_stats = self.dict_mesh_stats() 
                for _key,_value in _mesh_stats.items():
                    # print(f'output mesh stats {_key} {_value}')
                    _parameters["mesh_stats"][_key]=_value[_m]
                print(_parameters)
                afile._write_pickled_dict('/', 'parameters', _parameters)

                # save forward solution
                if 'steady' in mesh_seq.time_partition.field_types:
                    # QC:
                    print("\t checkpoint - saving: steady forward")
                    afile.save_function(
                        mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][0]
                        )
                else:
                    # QC:
                    print(f"\t checkpoint - saving: unsteady forward,\
                    range {np.shape(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'])[0]}")

                    for _t in range(np.shape(
                        mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'])[0]):
                        # it_sol = _t * mesh_seq.time_partition.num_timesteps_per_export[0] * mesh_seq.time_partition.timesteps[0]

                        # QC:
                        # print(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][_t].name())

                        afile.save_function(
                            mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][_t],
                            idx = _t
                        )

                # save adjoint - 
                # TODO: is saving the adjoint needed
                if not mesh_seq.steady: # is this the correct filter?
                    # QC:
                    print("\t checkpoint - saving: adjoint")

                    afile.save_function(
                        mesh_seq.solutions.extract(layout='subinterval')[_m][field]['adjoint'][0]
                        )

                # save indicators
                if mesh_seq.indicators is not None:
                    # QC:
                    print("\t checkpoint - saving: indicators")
                    afile.save_function(
                        mesh_seq.indicators.extract(layout='subinterval')[_m][field][0]
                        )

    def adapt_outputs(self, mesh_seq, method):

        # write out pickle of export mesh and ML parameters
        features = self.export_features_one_iter()

        # output vtks
        self.output_vtk_for(mesh_seq)
        if method not in ["uniform", "steady_hessian", "time_hessian"]:
            self.output_vtk_adj(mesh_seq)

        # output checkpoint
        self.output_checkpoint(mesh_seq)

        return features
      

    def adaptor(self, mesh_seq, solutions = None, indicators=None):

        self.adapt_iteration +=1
        print(f"\n adaptor iteration: {self.adapt_iteration}")

        method = mesh_seq.params["adaptor_method"] 
        
        # Extract features and Exports
        features = self.adapt_outputs(mesh_seq, method)

        # QC:
        print('\t adaptor - export initial features, vtk and checkpoint')


        # ML Surrogate for Indicators
        field = mesh_seq.params["field"]
        indimethod = mesh_seq.params["indicator_method"] 

        print(f"\t adaptor - adapt method: {method} , ml indicator?: {indimethod}")
        
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

        # has to be before adaptation so solution and mesh in same Space
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )        

        self.mesh_stats.append(self.dict_mesh_stats())

        # QC
        # print(f"\t adaptor - current mesh: {self.mesh_stats[-1]} elements")

        print(f'\t adaptor - calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, uniform')

        # timer start:
        duration = -perf_counter()

        # only run to min number of iterations
        if iteration >= mesh_seq.params["miniter"]:
            print(f'\n reached max uniform iterations at {mesh_seq.fp_iteration}\
             base on min iter {mesh_seq.params["miniter"]}')
            return False

        # Adapt 
        for i, mesh in enumerate(mesh_seq):
            
            # use mesh hierarchy to halve mesh spacing until converged
            if not mesh_seq.converged[i]:
                # QC:
                print(f'\t adaptor - new mesh name: mesh_{i}_{iteration+1}')
                mh = fd.MeshHierarchy(mesh_seq.meshes[i],1)
                mesh_seq.meshes[i] = mh[-1]
                mesh_seq.meshes[i].name = f'mesh_{i}_{iteration+1}'
            else:
                # TODO: is output stats post adaptation necessary
                pass
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(fd.assemble(mesh_seq.calc_qoi(-1,mesh_seq.solutions[field][label][-1][-1])))
                # QC:
                print("\t adaptor - final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")


        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, uniform')
        print(f'\t adaptor - uniform adaptor time taken: {duration:.6f} seconds')

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

        # paramters
        label='forward'
        field = mesh_seq.params["field"]
        iteration = mesh_seq.fp_iteration

        #fields
        solutions = mesh_seq.solutions       
     
        # FOR ANALYSIS
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )
        
        # QC
        print(f'\t adaptor - steady hessian, calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, steady, hessian')
        
        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        # ramp non-linear
        # TODO: move this to ramp function in goalie
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
            print(f"\t adaptor - steady hessian, solution loop i{i} ")

            # list to hold metrics per time step - to be combined later
            self.hessians=[]
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\t\t adaptor - steady hessian, solution loop j{j} ")
            
                # Recover the Hessian of the current solution
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                hessian = hessians[0] # ej321 - set the first hessian to the base
                hessian.average(*hessians[1:]) # ej321 - all the other hessians
                hessian.set_parameters(mp)
            
                # TODO: check if this is necesary  as steps not in original nn_adapt workflow
                # for steady state - space normalisation
                hessian.normalise()

                self.hessians.append(hessian)
                
            _hessians = self.hessians

            # constrain metric
            # QC: 
            print("\t adaptor - steady hessian: applying metric constraints")

            gol_adj.enforce_variable_constraints(_hessians, 
                h_min = mesh_seq.params["h_min"],
                h_max= mesh_seq.params["h_max"],
                a_max= mesh_seq.params["a_max"]
                )

            # QC:
            print(f'\t\t constrained metrics by\
             h_min {mesh_seq.params["h_min"]}\t\
             h_max {mesh_seq.params["h_max"]}\t\
             a_max {mesh_seq.params["a_max"]}')

            # set the first hessian to the base and average remaining
            metric =_hessians[0]
            metric.average(*_hessians[1:])
            
            # QC:
            # print(f'\t adaptor - steady hessian, combining {len(_metrics)} metrics for timestep')     
        
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
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                # QC:
                # print(f'\t adaptor - steady hessian, new mesh after adaptation: {mesh_seq.meshes[i].name}')
            else:
                # TODO: is output stats post adaptation necessary
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(fd.assemble(
                    mesh_seq.calc_qoi(
                        mesh_seq.calc_qoi(-1, solutions)
                        )))
                # QC:
                print("\t adaptor - steady hessian: final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, steady, hessian')
        print(f'\n\t adaptor - steady hessian: adaptor time taken: {duration:.6f} seconds')

        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        
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
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )

        print(f'\t adaptor - unsteady hessian, calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, time, hessian')

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
        
        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(solutions[field][label]):

            # QC:
            print(f"\t adaptor - unsteady hessian,, solution loop i{i} ")
            # print(f"- sols_step: {sols_step}")
            
            # time step
            dt = mesh_seq.time_partition.timesteps[i]

            # list to hold metrics per time step - to be combined later
            self.hessians=[]
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\t\t adaptor - unsteady hessian, solution loop j{j} ")
            
                # Recover the Hessian of the current solution
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                # set the first hessian to the base and average remaining
                hessian = hessians[0]
                hessian.average(*hessians[1:])
                hessian.set_parameters(mp)
            
                # TODO: check if this is necesary  as steps not in original nn_adapt workflow
                hessian.normalise()

                # append the metric for the step in the time partition
                # TODO: just output 
                self.hessians.append(hessian)

            _hessians = self.hessians

            # constrain metric
            # QC: 
            print("\t adaptor - unsteady hessian: applying metric constraints")
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

            # set the first hessian to the base and average remaining
            metric =_hessians[0]
            # TODO: should be equivalent to time integration
            metric.average(*_hessians[1:], weights=[dt]*len(_hessians))

            # QC:
            # print(f'\t adaptor - unsteady hessian, combining {len(_metrics)} metrics for timestep')     
        
            self.metrics.append(metric)
        
        # output metrics to vtk
        self.output_vtk_hes(mesh_seq, mesh_seq.params["adaptor_method"])
        self.output_vtk_met(mesh_seq, mesh_seq.params["adaptor_method"])
         
        metrics = self.metrics

        # QC:
        print(f'\t adaptor - unsteady hessian: before space-time normalisation \
        complexities: {[metric.complexity() for metric in metrics]}')

        # Apply space time normalisation
        gol_adj.space_time_normalise(metrics, mesh_seq.time_partition, mp)

        # QC:
        print(f'\t adaptor - unsteady hessian: after space-time normalisation \
        complexities: {[metric.complexity() for metric in metrics]}')

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                # QC:
                # print(f'\t adaptor - unsteady hessian, new mesh after adaptation: {mesh_seq.meshes[i].name}')
            else:
                # TODO: is output stats post adaptation necessary
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1, solutions)
                        )
                    )
                # QC:
                print("\t adaptor - unsteady hessian: final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, time, hessian')
        print(f'\n\t adaptor - unsteady hessian: adaptor time taken: {duration:.6f} seconds')

        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        
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
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )

        #QC
        print(f'\t adaptor - steady isotropic, calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, steady, isotropic')
        
        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        # ramp non-linear
        # TODO: move this to ramp function in goalie
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
            print(f"\t adaptor - steady isotropic, solution loop i{i} ")
            # print(f"- sols_step: {sols_step}")

            # list to hold metrics per time step - to be combined later
            _metrics=[]

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):
                
                # QC:
                print(f"\t\t adaptor - steady isotropic, solution loop j{i} ")
                # print(f"- sols_step: {sols_step}")

                # get local indicator 
                indi = indicators.extract(layout="field")[field][i][j] #update indicators
                # estimator = indi.vector().gather().sum()
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an isotropic metric from the error indicator field
                _metric.compute_isotropic_metric(error_indicator=indi, interpolant="L2")
                # TODO: check if this is necesary  as steps not in original nn_adapt workflow
                _metric.normalise()
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)

            # constrain metric
            # QC: 
            print("\t adaptor - steady isotropic: applying metric constraints")
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
            
            # set the first hessian to the base and average remaining
            metric =_metrics[0]
            metric.average(*_metrics[1:])
                        
            # QC:
            # print(f'\t adaptor - steady isotropic, combining {len(_metrics)} metrics for timestep')  
        
            self.metrics.append(metric)
            
        # output metrics to vtk
        self.output_vtk_met(mesh_seq)

        metrics = self.metrics
        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:

                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                # QC:
                # print(f'\t adaptor - steady isotropic, new mesh after adaptation: {mesh_seq.meshes[i].name}')

            else:
                # TODO: is output stats post adaptation necessary
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1, solutions)
                        )
                    )
                # QC:
                print("\t adaptor - steady isotropic: final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, steady, isotropic')
        print(f'\n\t adaptor - steady isotropic: adaptor time taken: {duration:.6f} seconds')
            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        
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
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )
        print(f'\t adaptor - unsteady isotropic, calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, time, isotropic')       
        
        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base = mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        
        # ramp non-linear
        # TODO: move this to ramp function in goalie
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
            print(f"\t adaptor - unsteady isotropic, solution loop i{i} ")
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

                # QC:
                print(f"\t\t adaptor - unsteady isotropic, solution loop j{i} ")
                
                # get local indicator 
                indi = indicators.extract(layout="field")[field][i][j] #update indicators
                # estimator = indi.vector().gather().sum()
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an isotropic metric from the error indicator field
                _metric.compute_isotropic_metric(error_indicator=indi, interpolant="L2")
                # TODO: check if this is necesary  as steps not in original nn_adapt workflow
                _metric.normalise()
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)
            
            # constrain metric
            # QC: 
            print("\t adaptor - unsteady isotropic: applying metric constraints")
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

            # set the first hessian to the base and average remaining
            metric =_metrics[0]
            metric.average(*_metrics[1:], weights=[dt]*len(_metrics))

            # QC:
            # print(f'\t adaptor - unsteady isotropic, combining {len(_metrics)} metrics for timestep') 
        
            # metrics.append(metric)
            self.metrics.append(metric)

        # output metrics to vtk
        self.output_vtk_met(mesh_seq, mesh_seq.params["adaptor_method"]) 

        metrics = self.metrics   

        # QC:
        print(f'\t adaptor - unsteady isotropic: before space-time normalisation\
         complexities: {[metric.complexity() for metric in metrics]}')

        # Apply space time normalisation
        gol_adj.space_time_normalise(metrics, mesh_seq.time_partition, mp)

        # QC:
        print(f'\t adaptor - unsteady isotropic: after space-time normalisation\
         complexities: {[metric.complexity() for metric in metrics]}')

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:

                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                # QC:
                # print(f'\t adaptor - unsteady isotropic, new mesh after adaptation: {mesh_seq.meshes[i].name}')
                
                #QC - for fixed mesh segments
                # print(f"\t adaptor - unsteady isotropic,face labels after adapt :\
                # {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

                self.unset_fixed_boundary(mesh_seq.meshes[i])
                self.unset_fixed_area(mesh_seq.meshes[i])

                # QC - for fixed mesh segments
                # print(f"\t adaptor - unsteady isotropic,face labels after adapt unset :\
                # {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

            else:
                # TODO: is output stats post adaptation necessary
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1,solutions)
                        )
                    )
                # QC:
                print("\t adaptor - unsteady isotropic: final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, time, isotropic')
        print(f' time, isotropic adaptor time taken: {duration:.6f} seconds')
            
            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        
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
        self.mesh_stats.append(self.dict_mesh_stats())
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )

        #QC
        print(f'\t adaptor - steady anisotropic, calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, steady, anisotropic')   

        # timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base = mesh_seq.params["base"]
        target = mesh_seq.params["target"]
        iteration = mesh_seq.fp_iteration

        # ramp non-linear
        # TODO: move this to ramp function in goalie
        # ramp_test=list(base+(target-np.geomspace(target,base,
        # num=mesh_seq.params["miniter"]+1, dtype=int))) +[target]*20
        
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

            # QC:
            print(f"\t adaptor - steady anisotropic, solution loop i{i} ")
                        
            # list to hold metrics per time step - to be combined later
            _metrics=[]
            _hessians=[]

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\t\t adaptor - steady anisotropic, solution loop j{i} ")
                
                # get local indicator 
                indi = indicators[field][i][j]
                # estimator = indi.vector().gather().sum()

                # Recover the Hessian of the current solution
                hes_duration = -perf_counter()
                hessians = [*get_hessians(sol, metric_parameters=mp)] # ej321
                
                
                # timer end
                hes_duration += perf_counter()
                # log time:
                logging.info(f'TIMING, hessian, {hes_duration:.6f}, steady, anisotropic')
                print(f'\t\t adaptor - steady anisotropic, hessian time taken: {hes_duration:.6f} seconds')
                
                # set the first hessian to the base and average remaining
                hessian = hessians[0]
                hessian.average(*hessians[1:])
                
                # append list for analysis
                self.hessians.append(hessian)
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an anisotropic metric from the error indicator field and the Hessian
                # TODO: pass interpolant as parameter
                # _metric.compute_anisotropic_dwr_metric(indi, hessian, interpolant="L2")
                _metric.compute_anisotropic_dwr_metric(indi, hessian) #default interpolant is Clement
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)

            #constrain metric
            # QC: 
            print("\t adaptor - steady anisotropic: applying metric constraints")
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

            # set the first hessian to the base and average remaining
            metric =_metrics[0]
            metric.average(*_metrics[1:])
            # QC:
            # print(f'\t adaptor - steady anisotropic, combining {len(_metrics)} metrics for timestep')  
            metric.normalise(restrict_sizes=False, restrict_anisotropy=False)
            metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True) # ej321 added


            self.metrics.append(metric)

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
                # QC:
                # print(f'\t adaptor - steady anisotropic, new mesh after adaptation: {mesh_seq.meshes[i].name}')
            else:
                # TODO: is output stats post adaptation necessary
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1, solutions)
                        )
                    )
                # QC:
                print("\t adaptor - steady anisotropic: final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, steady, anisotropic')
        print(f'\n\t adaptor - steady anisotropic: adaptor time taken: {duration:.6f} seconds')

            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)
        
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
        
        # FOR ANALYSIS
        self.mesh_stats.append(self.dict_mesh_stats()) 
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, solutions)
                )
            )

        #QC
        print(f'\t adaptor - unsteady anisotropic, calculated qoi: {self.qois[-1]}\n')
        logging.info(f'QOI, {self.qois[-1]}, time, anisotropic') 
        
        #timer start:
        duration = -perf_counter()

        # Ramp the target average metric complexity per timestep
        base =mesh_seq.params["base"]
        target = mesh_seq.params["target"]

        # ramp non-linear
        # TODO: move this to ramp function in goalie
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
            print(f"\t adaptor - unsteady anisotropic, solution loop i{i} ")
            # print(f" - sol: {sol}")

            dt = mesh_seq.time_partition.timesteps[i]
                    
            # list to hold metrics per time step - to be combined later
            _metrics=[]
            _hessians=[]

            # QC:
            print('\t adaptor - unsteady anisotropic, set fixed boundary/area')
            self.set_fixed_boundary(mesh_seq.meshes[i])
            self.set_fixed_area(mesh_seq.meshes[i])

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            metric = ani.metric.RiemannianMetric(P1_ten)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):

                # QC:
                print(f"\t adaptor - unsteady anisotropic, solution loop j{j} ")
                # print(f" - sol: {sol}")
                
                # get local indicator 
                indi = indicators[field][i][j]
                # estimator = indi.vector().gather().sum()

                # Recover the Hessian of the current solution
                hessians = [*get_hessians(sol, metric_parameters=mp)]
                # set the first hessian to the base and average remaining
                hessian = hessians[0]
                hessian.average(*hessians[1:])

                hessian.set_parameters(mp) # needed here?

                self.hessians.append(hessian)
                                    
                # local instance of Riemanian metric
                _metric = ani.metric.RiemannianMetric(P1_ten)
                # reset parameters as a precaution
                _metric.set_parameters(mp)
                
                # Deduce an anisotropic metric from the error indicator field
                # TODO: pass interpolant as parameter
                _metric.compute_anisotropic_dwr_metric(indi, hessian, interpolant="L2")
                
                # normalise the local metric
                # TODO: check if this is the right spot for this
                _metric.normalise()
                
                # append the metric for the step in the time partition
                _metrics.append(_metric)           
            
            #constrain metric
            # QC: 
            print("\t adaptor - unsteady anisotropic: applying metric constraints")
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

            # set the first metric to the base and average remaining
            metric =_metrics[0]
            metric.average(*_metrics[1:], weights=[dt]*len(_metrics))

            # QC:
            # print(f'\t adaptor - unsteady anisotropic, combining {len(_metrics)} metrics for timestep')
  
            # metrics.append(metric)
            self.metrics.append(metric)

            # output metrics to vtk
            self.output_vtk_hes(mesh_seq)
            self.output_vtk_met(mesh_seq)
            self.output_vtk_ind(mesh_seq)

            metrics = self.metrics

        ##*****************************************************************************************##
        
        # QC:
        print(f'\t adaptor - unsteady anisotropic: before space-time normalisation\
         complexities: {[metric.complexity() for metric in metrics]}')

        # Apply space time normalisation
        gol_adj.space_time_normalise(metrics, mesh_seq.time_partition, mp)

        # QC:
        print(f'\t adaptor - unsteady anisotropic: after space-time normalisation\
         complexities: {[metric.complexity() for metric in metrics]}')

        # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
        for i, metric in enumerate(metrics):
            
            # re-estimate resulting metric complexity 
            complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i], metric, name=f'mesh_{i}_{iteration+1}')
                # QC:
                # print(f'\t adaptor - unsteady anisotropic, new mesh after adaptation: {mesh_seq.meshes[i].name}')
                
                # QC:
                print('\t adaptor - unsteady anisotropic: unseting fixed boundary/areas')
                # print(f"\t\t face labels after adapt : {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

                self.unset_fixed_boundary(mesh_seq.meshes[i])
                self.unset_fixed_area(mesh_seq.meshes[i])

                # QC:
                # print(f"\t\t\ face labels after adapt uset : {mesh_seq.meshes[i].topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices}")

            else:
                # TODO: is output stats post adaptation necessary
                self.mesh_stats.append(self.dict_mesh_stats()) 
                self.qois.append(
                    fd.assemble(
                        mesh_seq.calc_qoi(-1,solutions[field][label][-1][-1])
                        )
                    )
                # QC:
                print("\t adaptor - unsteady anisotropic: final mesh stats exported")
                print(f"\n\t final mesh stats: {self.mesh_stats[-1]}")

        # timer end
        duration += perf_counter()
        # log time:
        logging.info(f'TIMING, adaptor, {duration:.6f}, time, anisotropic')
        print(f' time, anisotropic adaptor time taken: {duration:.6f} seconds')
            
            
        # update metrics 
        num_dofs = mesh_seq.count_vertices()
        num_elem = mesh_seq.count_elements()
        
        self.complexities.append(complexities)

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