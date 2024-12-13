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
from workflow.timer import Timer


from collections import defaultdict
import abc

# def output_checkpoint(self, mesh_seq, field = None, ):

#     if field is None:
#         field = mesh_seq.params["field"]

#     # write out Checkpoint
#     # QC:
#     print(f'Writing Checkpoint: {mesh_seq.meshes}, {mesh_seq.time_partition.field_types}') 
#     for _m, _mesh in enumerate(mesh_seq):
#         # QC:
#         print(f'\t checkpoint - saving: {_m} {_mesh} {_mesh.name} ')
#         chkpt_file = os.path.join(os.getcwd(),
#                 f'_{field}_{mesh_seq.params["adaptor_method"]}_mesh_{_m}_{mesh_seq.fp_iteration}.h5'
#                     )
#         with fd.CheckpointFile(chkpt_file, 'w') as afile:
#             afile.save_mesh(_mesh)
#             print(f'output params: {type(dict(mesh_seq.params))}')
#             _parameters={
#                 "params": dict(mesh_seq.params), # ej321 test
#                 "mesh_stats":{} 
#             }
            
#             # _parameters = mesh_seq.params.to_dict() # convert AttrDict back to dict type
#             # _parameters["mesh_stats"]={}
#             # add mesh stats into the general stats dictionary
#             _mesh_stats = self.dict_mesh_stats() 
#             for _key,_value in _mesh_stats.items():
#                 # print(f'output mesh stats {_key} {_value}')
#                 _parameters["mesh_stats"][_key]=_value[_m]
#             print(_parameters)
#             afile._write_pickled_dict('/', 'parameters', _parameters)

#             # save forward solution
#             if 'steady' in mesh_seq.time_partition.field_types:
#                 # QC:
#                 print("\t checkpoint - saving: steady forward")
#                 afile.save_function(
#                     mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][0]
#                     )
#             else:
#                 # QC:
#                 print(f"\t checkpoint - saving: unsteady forward,\
#                 range {np.shape(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'])[0]}")

#                 for _t in range(np.shape(
#                     mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'])[0]):
#                     # it_sol = _t * mesh_seq.time_partition.num_timesteps_per_export[0] * mesh_seq.time_partition.timesteps[0]

#                     # QC:
#                     # print(mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][_t].name())

#                     afile.save_function(
#                         mesh_seq.solutions.extract(layout='subinterval')[_m][field]['forward'][_t],
#                         idx = _t
#                     )

#             # save adjoint - 
#             # TODO: is saving the adjoint needed
#             if not mesh_seq.steady: # is this the correct filter?
#                 # QC:
#                 print("\t checkpoint - saving: adjoint")

#                 afile.save_function(
#                     mesh_seq.solutions.extract(layout='subinterval')[_m][field]['adjoint'][0]
#                     )

#             # save indicators
#             if mesh_seq.indicators is not None:
#                 # QC:
#                 print("\t checkpoint - saving: indicators")
#                 afile.save_function(
#                     mesh_seq.indicators.extract(layout='subinterval')[_m][field][0]
#                     )


def AdaptorSelection(method = "uniform", **kwargs):
    adaptor_methods = {
        "uniform": Adaptor_B,
        "hessian": Adaptor_H,
        "isotropic": Adaptor_I,
        "anisotropic": Adaptor_A,
    }
    try:
        return adaptor_methods[method](**kwargs)
    except KeyError as e:
        raise ValueError(f"Method '{method}' not recognised.") from e

class Adaptor_B(metaclass =abc.ABCMeta):
    """
    Base class for Adaptors
    """
    def __init__(self, **kwargs):
        self.field = kwargs.get("field")
        self.local_filepath = kwargs.get("local_filepath")
        self.adaptor_method = kwargs.get("adaptor_method")
        self.adapt_iteration = 0
        self.qois = []
        self.mesh_stats = []
    

    def _output_options(self):
        """
        Return a dictionary of options for file outputs
        """
        return {
            "forward": self._output_forward,
            "adjoint": self._output_adjoint,
        }

    def _output_selection(self, output_list=[], mesh_seq = None, format="vtk", **kwargs):
        """
        Sets up field exports for adaptor related fields
        TODO: expand for more than 'vtk' files
        """
        
        output_options = self._output_options()

        if not output_list:
            output_list= output_options.keys()

        vtk_folder = f"vtk_files_fpi{self.adapt_iteration}"

        for output in output_list:
            file_out= VTKFile(f"{vtk_folder}/{output}.pvd")
            try:
                output_options[output](file_out,mesh_seq=mesh_seq, **kwargs)
                print(f"output item - {output}")
            except KeyError as e:
                raise ValueError(f"Output '{output}' not currently defined.") from e


    def _output_forward(self, file_out, mesh_seq):
        """
        Output forward solution to vtk, either for steady or unsteady
        sets the output file and writes to it
        """

        assert self.field, f"field value must be set"
        assert mesh_seq, f"MeshSeq object is not set"
        _sol_obj =  mesh_seq.solutions.extract(layout='subinterval')

        assert all([True for _sol in _sol_obj if _sol[self.field]["forward"]]),\
        f"issue with forward solution field"

        for _sol in _sol_obj:
            for _t in range(np.shape(_sol[self.field]["forward"])[0]):
                file_out.write(*_sol[self.field]['forward'][_t].subfunctions)


    def _output_adjoint(self, file_out, mesh_seq):
        """
        Output adjoint solution to vtk, either for steady or unsteady
        sets the output file and writes to it
        """

        assert self.field, f"field value must be set"
        assert mesh_seq, f"MeshSeq object is not set"
        _sol_obj =  mesh_seq.solutions.extract(layout='subinterval')

        assert all([True for _sol in _sol_obj if _sol[self.field]["adjoint"]]),\
        f"issue with adjoint solution field"

        for _sol in _sol_obj:
            for _t in range(np.shape(_sol[self.field]["adjoint"])[0]):
                # u.rename(name = 'elev_2d')
                # uv.rename(name = 'uv_2d')
                file_out.write(*_sol[self.field]['adjoint'][_t].subfunctions)


    def dict_mesh_stats(self, mesh_seq):
        """
        Computes and returns a dictionary containing statistics for each mesh in the sequence.

        This method iterates over a sequence of mesh objects and collects statistics including
        the number of elements and vertices, as well as the mean, minimum, and maximum values 
        for each quality metric.

        :returns: summary mesh statictics in a dictionary

        """
        
        def_dict = defaultdict(list)
        for m in mesh_seq:
            def_dict['elements'].append(m.num_cells())
            def_dict['vertices'].append(m.num_vertices())

            # TODO: 
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
    
    def plot_mesh_convergence_stats(self, mesh_seq, plot_len=10, subplot_ht=2, show=False):
        
        # these are repeated so just grabbing the first one
        def_dict = self.mesh_stats[0]

        # use the length of the first one to create plots for each
        fig,ax = plt.subplots(len(def_dict)+2, 1,
                figsize=(plot_len,subplot_ht*len(def_dict)),
                sharex=True, layout="constrained")

        # iterate over each statitic category and get key name
        ax[0].plot(mesh_seq.estimator_values)
        ax[0].set_ylabel("estimator value")
        qoi_vals = max(self.qois,mesh_seq.qoi_values)
        ax[1].plot(qoi_vals)
        ax[1].set_ylabel("qoi value")
        plt.xlabel("iterations")
        for d,key in enumerate(self.mesh_stats[0].keys()): # d in mesh statistic
            
            # get the number of meshes - reused so extract to variable once
            nmesh=len(mesh_seq.meshes)
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
                f'mesh_convergence_statistics.jpg'
                )
        print(f"fig saving to: {_filepath}")
        fig.savefig(_filepath)

        if show:
            plt.show()


    def _mesh_stats(self, mesh_seq):
            """
            extract general mesh statistics for output
            TODO: currently only works for one mesh steady state only
            """
            gen_stats={}
            # gen_stats["params"]=self.params # not currently passed
            gen_stats["mesh_id"] = 0
            gen_stats["fp_iteration"] = mesh_seq.fp_iteration

            gen_stats["dof"]=sum(np.array([
                mesh_seq.solutions[self.field]['forward'][0][0]
                .function_space().dof_count
                ]).flatten())

            _mesh_stats = self.dict_mesh_stats(mesh_seq) 
            for _key,_value in _mesh_stats.items():
                gen_stats[_key]=_value[0]

            return gen_stats
    
    def _mesh_features(self, mesh_seq):
        """
        extract some derivative parameters from the mesh
        TODO: currently only works for one mesh steady state only
        """
        fwd_sol = mesh_seq.solutions[self.field]['forward'][0][0]
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

        return features

    def _model_features(self, mesh_seq):
        """
        return model parameters extracted from the mesh solve
        TODO: currently only works for one mesh steady state only
        """
        features = {}
        try:
            for _key,_value in mesh_seq.model_features.items():
                # print(f'in feature output: {_key,_value}')
                features[_key] = _value
        except:
            print("Issue with accessing model features on Model object - did you mean to define?")

        return features


    def _field_features(self, mesh_seq):
        """
        return field related feature parameters
        TODO: currently only works for one mesh steady state only
        """
        features = {}
        features["forward_dofs"]={}
        fwd_sol = mesh_seq.solutions[self.field]['forward'][0][0]
        for _t in range(np.shape(mesh_seq.solutions[self.field]['forward'][0])[0]):
                it_sol = _t * mesh_seq.time_partition.num_timesteps_per_export[0] * mesh_seq.time_partition.timesteps[0]
                it_sol = str(round(it_sol,2))
                print(it_sol)
                
                features["forward_dofs"][it_sol] = extract_array(
                    mesh_seq.solutions[self.field]['forward'][0][_t],centroid=True) 

        # if adjoint run - extract adjoint and estimator
        if 'adjoint' in mesh_seq.solutions[self.field]:
            try:
                adj_sol = mesh_seq.solutions[self.field]['adjoint'][0][0]
                # print(f'EJ321 - adj in feature export {adj_sol}')
                features["adjoint_dofs"] = extract_array(adj_sol, centroid=True)
                # print(f'EJ321 - adj in adj_dofs {features["adjoint_dofs"]}')
                features["estimator_coarse"] = extract_coarse_dwr_features(mesh_seq, fwd_sol, adj_sol, index=0)
                # print(f'EJ321 - adj in estimator_coarse {features["estimator_coarse"]}')
            except:
                print(f'issue with extracting adjoint')
        
        return features

    @Timer(logger = logging.info, text="adaptor - get all features, time taken: {:.6f}")
    def _get_all_features(self, mesh_seq):
        """
        get all features for export
        """
        features = {}
        features['gen_stats'] = self._mesh_stats(mesh_seq)
        features.update(self._mesh_features(mesh_seq))
        features.update(self._model_features(mesh_seq))
        features.update(self._field_features(mesh_seq))

        return features


    def _export_features(self, mesh_seq, features):
        """
        export features for ML model input
        TODO: currently only works for one mesh steady state only
        """

        output_file = f'{self.field}_{self.adaptor_method }_mesh_0_{mesh_seq.fp_iteration}.pkl'

        with open(
            output_file,'wb') as pickle_file:
                pickle.dump(features, pickle_file)

    @Timer(logger = logging.info, text="adaptor - get inital mesh stats, time taken: {:.6f}")
    def _get_initial_stats(self, mesh_seq):
        """
        append statistics to relavent lists
        """
        self.qois.append(
            fd.assemble(
                mesh_seq.calc_qoi(-1, mesh_seq.solutions)
                )
            )
        self.mesh_stats.append(self.dict_mesh_stats(mesh_seq))
        
        # log stats
        logging.info(f'qoi: {self.qois[-1]}')
        logging.info(f'mesh_stats: {self.mesh_stats[-1]}')

        # other:

        # timer start

    # metric loop
    def _calculate_metrics(self, mesh_seq):
        pass
    

    # adapt the mesh
    @Timer(logger = logging.info, text="adaptor - mesh adapt time taken: {:.6f}")
    def _adapt_meshes(self, mesh_seq):

        for i in range(len(mesh_seq)):
            # only run to min number of iterations
            # TODO: check where parsing miniter from now
            if mesh_seq.fp_iteration >= mesh_seq.params["miniter"]:
                print(f'\n reached max iterations at {mesh_seq.fp_iteration}\
                base on min iter {mesh_seq.params["miniter"]}')
                return False
        
            # use mesh hierarchy to halve mesh spacing until converged
            if not mesh_seq.converged[i]:
                # QC:
                print(f'\t adaptor - new mesh name: mesh_{i}_{mesh_seq.fp_iteration +1}')
                mh = fd.MeshHierarchy(mesh_seq.meshes[i],1)
                mesh_seq.meshes[i] = mh[-1]
                mesh_seq.meshes[i].name = f'mesh_{i}_{mesh_seq.fp_iteration +1}'               
        

    # print info
    def _adaptor_info(self, mesh_seq):
        """
        Print progress of adaptation to screen
        """
        num_vertices = mesh_seq.count_vertices()
        num_elements = mesh_seq.count_elements()
        gol_adj.pyrint(f'adaptor - finish iteration: {mesh_seq.fp_iteration + 1}')
        # gol_adj.pyrint(f"adapt iteration {mesh_seq.fp_iteration + 1}:")
        for i, (nvert, nelem) in enumerate(zip(num_vertices, num_elements)):
            gol_adj.pyrint(
                f"subinterval: {i} vertices: {nvert:4d} elements: {nelem:4d}"
            )
    
    @Timer(logger = logging.info, text="adaptor - adaptor, time taken: {:.6f}")
    def adaptor(self, mesh_seq, solutions = None, indicators=None):
        self.adapt_iteration +=1
        features = self._get_all_features(mesh_seq)
        self._export_features(mesh_seq, features)
        # TODO: output checkpoint file
        self._get_initial_stats(mesh_seq)
        self._output_selection(mesh_seq = mesh_seq, format="vtk")
        self._calculate_metrics(mesh_seq)
        self._adapt_meshes(mesh_seq)
        self._adaptor_info(mesh_seq)
        return True


class Adaptor_H(Adaptor_B):
    """
    Hessian class for Adaptors
    """
    def __init__(self, **kwargs):
        self.miniter = kwargs.get("miniter")
        self.maxiter = kwargs.get("maxiter")
        self.base_complexity = kwargs.get("base")
        self.target_complexity = kwargs.get("target")
        self.h_min = kwargs.get("h_min")
        self.h_max = kwargs.get("h_max")
        self.a_max = kwargs.get("a_max")
        self.complexities = []
        self.ramp_complexities = []
        self.hessians=[]
        self.metrics=[]
        super().__init__(**kwargs)


    def _output_options(self):
        """
        Return a dictionary of options for file outputs
        """
        return {
            "forward": self._output_forward,
            "adjoint": self._output_adjoint,
            "hessian": self._output_hessian,
            "metric": self._output_metric
        }

    def _output_hessian(self, file_out, **kwargs):
        """
        Output hessian to vtk, either for steady or unsteady
        sets the output file and writes to it
        """

        assert self.hessians, f"issue with hessian field"

        #TODO: add check hessian and mesh_seq are consistent

        for _hes in self.hessians:
            _hes.rename(f'hessian')
            file_out.write(*_hes.subfunctions)


    def _output_metric(self, file_out, **kwargs):
        """
        Output metric to vtk, either for steady or unsteady
        sets the output file and writes to it
        """

        assert self.metrics, f"issue with metric field"
        
        #TODO: add check metric and mesh_seq are consistent

        for _met in self.metrics:
            _met.rename(f'metric')
            file_out.write(*_met.subfunctions)

    @Timer(logger = logging.info, text="adaptor - set ramp complexity, time taken: {:.6f}")
    def _set_ramp_complexity(self):
        """
        function to set the ramp complexity based on base and targets set 
        """
        ramp_complexity_list=list(self.base_complexity+(self.target_complexity
                                             -np.geomspace(self.target_complexity,
                                                 self.base_complexity, 
        num=self.miniter+1, dtype=int))) +[self.target_complexity]*20
        

        return ramp_complexity_list

    def _set_metric_parameters(self, mesh_seq):
        """
        set the metric parameters to be used for emtric construction
        """
        
        ramp_complexity_list = self._set_ramp_complexity()
        
        metric_params = {"dm_plex_metric_target_complexity":ramp_complexity_list[mesh_seq.fp_iteration],
                "dm_plex_metric_p": 1,
            }
        self.ramp_complexities.append(metric_params["dm_plex_metric_target_complexity"])

        return metric_params
    
    @Timer(logger = logging.info, text="adaptor - calculate hessian, time taken: {:.6f}")
    def _recover_hessian(self, solution, metric_parameters, average=True, normalise = True):
        hessians = [*get_hessians(solution, metric_parameters=metric_parameters)] # ej321
        hessian = hessians[0] # ej321 - set the first hessian to the base
        if average:
            hessian.average(*hessians[1:]) # ej321 - all the other hessians

        hessian.set_parameters(metric_parameters)
        
        # ej321 - steps not in Joe's workflow?
        # for steady state - space normalisation
        if normalise:
            hessian.normalise()

        return hessian
    
        

    @Timer(logger = logging.info, text="adaptor - calculate metric, time taken: {:.6f}")
    def _calculate_metrics(self, mesh_seq):

        self.metrics = []

        mp = self._set_metric_parameters(mesh_seq)

        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(mesh_seq.solutions[self.field]['forward']):

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
                hessian = self._recover_hessian(sol, metric_parameters=mp) # ej321
                
                # _metric = hessian

                # append the metric for the step in the time partition
                # _metrics.append(_metric)
                # TODO: just output 
                self.hessians.append(hessian)
                
            _hessians = self.hessians

            # constrain metric
            print(f"\n\n ADDING CONSTRAINS IN ADAPT HESSIAN STEADY: \n")
            gol_adj.enforce_variable_constraints(_hessians, 
                h_min = self.h_min,
                h_max= self.h_max,
                a_max= self.a_max
                )

            # QC:
            print(f'constrained metrics by\
             h_min: {self.h_min} h_max: {self.h_max} a_max: {self.a_max}')

            # print(f'combining {len(_metrics)} metrics for timestep')
            metric =_hessians[0] # ej321 - set the first hessian to the base

            if mesh_seq.steady:
                # TODO: does this actually do anything in the steady case?
                print(f'in steady hessian - hessian length {len(_hessians)}')
                metric.average(*_hessians[1:]) # ej321 - all the other hessians
            else:
                dt = mesh_seq.time_partition.timesteps[i]
                metric.average(*_hessians[1:], weights=[dt]*len(_hessians)) # ej321 - all the other hessians     
        
        
            # metrics.append(metric)
            self.metrics.append(metric)
                    
  

    # adapt the mesh
    @Timer(logger = logging.info, text="adaptor - adapt mesh, time taken: {:.6f}")
    def _adapt_meshes(self, mesh_seq):

        self.complexities = []

        if not mesh_seq.steady:
            # Apply space time normalisation
            mp = self._set_metric_parameters(mesh_seq)
            gol_adj.space_time_normalise(self.metrics, mesh_seq.time_partition, mp)

        for i, metric in enumerate(self.metrics):
                    
            # re-estimate resulting metric complexity 
            self.complexities.append(metric.complexity())
            
            # Adapt the mesh
            if not mesh_seq.converged[i]:
                # print(f'new mesh name mesh_{i}_{iteration+1}')
                mesh_seq.meshes[i] = ani.adapt(mesh_seq.meshes[i],
                                                metric, name=f'mesh_{i}_{mesh_seq.fp_iteration+1}')
                # print(f'new mesh: {mesh_seq.meshes[i].name}')

    def _adaptor_info(self, mesh_seq):
        """
        Print progress of adaptation to screen
        """
        num_vertices = mesh_seq.count_vertices()
        num_elements = mesh_seq.count_elements()
        iter_complex = self.ramp_complexities[mesh_seq.fp_iteration]
        gol_adj.pyrint(f'adaptor - finish iteration: {mesh_seq.fp_iteration + 1} target: {iter_complex}')
        # gol_adj.pyrint(f"adapt iteration {mesh_seq.fp_iteration + 1}:")
        for i, (nvert, nelem, complexity) in enumerate(zip(num_vertices, num_elements, self.complexities)):
            gol_adj.pyrint(
                f"subinterval: {i} vertices: {nvert:4d} elements: {nelem:4d} complexity: {complexity:4.0f}"
            )

    @Timer(logger = logging.info, text="adaptor - adaptor, time taken: {:.6f}")
    def adaptor(self, mesh_seq, solutions = None, indicators=None):
        self.adapt_iteration +=1
        features = self._get_all_features(mesh_seq)
        self._export_features(mesh_seq, features)
        self._get_initial_stats(mesh_seq)
        self._calculate_metrics(mesh_seq)
        self._output_selection(mesh_seq = mesh_seq, format="vtk")
        self._adapt_meshes(mesh_seq)
        self._adaptor_info(mesh_seq)

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(self.complexities) < 0.90 * self.target_complexity
        return continue_unconditionally

class Adaptor_I(Adaptor_H):
    """
    Isotropic class for Adaptors
    """
    def __init__(self, **kwargs):
        self.indicator_method = kwargs.get("indicator_method")

        super().__init__(**kwargs)

    def _output_options(self):
        """
        Return a dictionary of options for file outputs
        """
        return {
            "forward": self._output_forward,
            "adjoint": self._output_adjoint,
            "indicator": self._output_indicator,
            "metric": self._output_metric
        }
    
    def _output_indicator(self, file_out, mesh_seq):
        """
        Output indicator field to vtk, either for steady or unsteady
        sets the output file and writes to it
        """

        assert self.field, f"field value must be set"
        assert mesh_seq, f"MeshSeq object is not set"

        _ind_obj = mesh_seq.indicators.extract(layout='subinterval')
        if _ind_obj:
            # output the indicator function
            for _ind in _ind_obj:
                for _t in range(np.shape(_ind[self.field])[0]):
                    file_out.write(*_ind[self.field][_t].subfunctions)


    def _get_indicator(self, mesh_seq, features):
        """
        Get indicators
        """

        print(f"\n\n adaptor using: {self.adaptor_method} - { self.indicator_method }")
        indicator_methods = {
        "gnn": gnn_indicator_fit,
        "gnn_noadj": gnn_noadj_indicator_fit,
        "mlp": mlp_indicator_fit,
        }
        if self.indicator_method in indicator_methods:
             mesh_seq.indicators[self.field][0][0]= indicator_methods[self.indicator_method](
                 features, mesh_seq.meshes[-1])
             
        else:
            print(f'ML indicator method not selected')


    @Timer(logger = logging.info, text="adaptor - calculate metric, time taken: {:.6f}")
    def _calculate_metrics(self, mesh_seq):

        self.metrics = []

        mp = self._set_metric_parameters(mesh_seq)

        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(mesh_seq.solutions[self.field]['forward']):

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
                indi = mesh_seq.indicators.extract(layout="field")[self.field][i][j] #update indicators
                                    
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
            gol_adj.enforce_variable_constraints(_metrics, 
                h_min = self.h_min,
                h_max= self.h_max,
                a_max= self.a_max
                )

            # QC:
            print(f'constrained metrics by\
             h_min: {self.h_min} h_max: {self.h_max} a_max: {self.a_max}')
                
            metric =_metrics[0] # ej321 - set the first hessian to the base
            
            
            if mesh_seq.steady:
                metric.average(*_metrics[1:]) # ej321 - all the other hessians
            else:
                dt = mesh_seq.time_partition.timesteps[i]
                metric.average(*_metrics[1:], weights=[dt]*len(_metrics)) # ej321 - all the other hessians  
             
        
            # metrics.append(metric)
            self.metrics.append(metric)

        # OUTPUT TO VTK - CALLBACK? or in adaptor?
    @Timer(logger = logging.info, text="adaptor - adaptor, time taken: {:.6f}")
    def adaptor(self, mesh_seq, solutions = None, indicators=None):
        self.adapt_iteration +=1
        features = self._get_all_features(mesh_seq)
        self._export_features(mesh_seq, features)
        self._get_indicator(mesh_seq, features)
        self._get_initial_stats(mesh_seq)
        self._calculate_metrics(mesh_seq)
        self._output_selection(mesh_seq = mesh_seq, format="vtk")
        self._adapt_meshes(mesh_seq)
        self._adaptor_info(mesh_seq)

        # check if the target complexity has been (approximately) reached on each subinterval
        continue_unconditionally = np.array(self.complexities) < 0.90 * self.target_complexity
        return continue_unconditionally


class Adaptor_A(Adaptor_I):
    """
    Anisotropic class for Adaptors
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _output_options(self):
        """
        Return a dictionary of options for file outputs
        """
        return {
            "forward": self._output_forward,
            "adjoint": self._output_adjoint,
            "indicator": self._output_indicator,
            "hessian": self._output_hessian,
            "metric": self._output_metric
        }

    @Timer(logger = logging.info, text="adaptor - calculate metric, time taken: {:.6f}")
    def _calculate_metrics(self, mesh_seq):

        self.metrics = []

        mp = self._set_metric_parameters(mesh_seq)

        # Loop through each time step in the MeshSeq
        for i, sols_step in enumerate(mesh_seq.solutions[self.field]['forward']):

            # list to hold metrics per time step - to be combined later
            _metrics=[]

            # Define the Riemannian metric
            P1_ten = fd.TensorFunctionSpace(mesh_seq.meshes[i], "CG", 1)
            
            # loop through time increment solution in the current time step
            for j, sol in enumerate(sols_step):
                
                # get local indicator 
                indi = mesh_seq.indicators[self.field][i][j]

                # Recover the Hessian of the current solution
                hessian = self._recover_hessian(sol, metric_parameters=mp, normalise=False) # ej321
                
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

            # constrain metric
            gol_adj.enforce_variable_constraints(_metrics, 
                h_min = self.h_min,
                h_max= self.h_max,
                a_max= self.a_max
                )

            # QC:
            print(f'constrained metrics by\
             h_min: {self.h_min} h_max: {self.h_max} a_max: {self.a_max}')

            metric =_metrics[0] # ej321 - set the first hessian to the base
            if mesh_seq.steady:
                metric.average(*_metrics[1:]) # ej321 - all the other hessians
            else:
                dt = mesh_seq.time_partition.timesteps[i]
                metric.average(*_metrics[1:], weights=[dt]*len(_metrics)) # ej321 - all the other hessians  
             
            self.metrics.append(metric)


            # OUTPUT TO VTK - CALLBACK? or in adaptor?


if __name__ == "__main__":
    solution = "solution"
    indicator = "indicator"
    # adapt_test = Adaptor(solution)
    # print(adapt_test.solution, adapt_test.indicator)
    # final_solution =mesh_seq.__class__.mro()[3].fixed_point_iteration(mesh_seq,adaptor)
    # mesh_seq.solutions.append(final_solution[0])