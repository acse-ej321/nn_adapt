import firedrake as fd
import animate as ani # new sept23
# import goalie as gol # new sept23
import goalie_adjoint as gol_adj # new sept23
from workflow.adaptor import *

# from models import *
from workflow.features import *
# from workflow.utility import *
import os 
from datetime import datetime
from firedrake.petsc import PETSc
# import pdb; pdb.set_trace()

# timing functions manually:
import logging
from time import perf_counter

import shutil

class AttrDict(dict):
    """
    Dictionary that provides both ``self[key]`` and ``self.key`` access to members.

    **Disclaimer**: Copied from `stackoverflow
    <http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python>`__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def create_folder(rootfolder, folderkey, filepath):
    """
    Check if a folder exists and if not, create it
    
    Args:
        folder_name (string - directory)
    """
    foldername = str(folderkey) + datetime.now().strftime('%d_%m_%Y')
    try:
        if os.path.isdir(filepath):
            print(f'simulation using folder: {filepath}')
            return filepath
            
    except:
        # filepath = os.getcwd()
        filepath = rootfolder

        if not os.path.isdir(os.path.join(filepath,foldername)):
            # ej321 - add MPI for running on HPC
            # if MPI.COMM_WORLD.rank == 0:
            # print('Start time:', starttime.strftime("%d/%m/%Y %H:%M:%S"))
            os.makedirs(os.path.join(filepath,foldername)) 

            filepath = os.path.join(filepath,foldername)
            print(f'simulation folder created: {filepath}') 
            return filepath
        else:
            filepath = os.path.join(filepath,foldername)
            print(f'simulation using folder: {filepath}')
            return filepath

class Simulation():
    
    def __init__(self, Model, rootfolder=None, filepath=None, parameters={}):
        self.rootfolder = rootfolder if rootfolder else os.getcwd()
        self.filepath = create_folder(self.rootfolder, str(Model.__name__), filepath)
        # self.params = gol.GoalOrientedAdaptParameters() # not really worth it now only fpi ones included
        self.params=AttrDict() # TODO: set some defaults for all parameters passed
        self.update_params(Model.get_default_parameters())
        self.update_params(parameters)
        # print(f" simulation parameters: (self.params)") # TODO: comment out or add to log
        
        self.initial_meshes = Model.get_default_meshes(self.filepath,**self.params)
        self.time_partition = Model.get_default_partition(**self.params)
        self.mesh_seq = Model(self.time_partition,
                            self.initial_meshes, 
                                qoi_type=self.params["qoi_type"],
                                parameters=self.params # ej321 - this is not split from mesh_seq, so passing for child class
                                )
        
        # self.timer = TimeIt("timesout")
        self.adaptor=Adaptor(self.mesh_seq, self.params, filepath = self.filepath)
        
        # add path to mesh object
        # self.mesh_seq.filepath = self.filepath # not needed? try commenting out
        

        # logging.FileHandler(os.path.join(self.filepath,"timesout.txt"))


    def setup_logging_file(self, filepath):
        # need to remove logging handler explicity to switch filepath if one has
        # previously initialised
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO, 
            filename=os.path.join(filepath,"timesout.txt"),
            format=' %(asctime)s - %(levelname)s - %(message)s'
            )



    def update_params(self, other_parameters):
        """
        for updating parameters with additional 
        
        adding feature from Joe's code parameter class to pass keys as methods
        TODO: figure out a better place for this
        """


        self.params.update(other_parameters)

    # def update_meshes(self, meshes, meshloc=None):
    #     print("update meshes")
    #     if meshloc is None:
    #         self.meshes = meshes
    #     else:
    #         self.meshes[meshloc] = meshes

    # def reload_default()

    # @TimeIt.timeit
    @PETSc.Log.EventDecorator("SIMULATION")
    def run_fp_loop(self):

        # print(self.filepath)
        
        # set output folder with curernt date
        self.adaptor.set_outfolder(f'{datetime.now().strftime("%H%M%S")}')
        self.setup_logging_file(self.adaptor.local_filepath)
        logging.info(f'\n################################################')
        logging.info(f'\n\tFIXED_POINT_ITERATION_FOLDER: {os.getcwd()}')
        logging.info(f'\n parameters: {self.params}')
        
        # timer start:
        duration = -perf_counter()
        # set up folder in the parent simulation folder for interim outputs


        # QC base class has both solution and indicators
        self.mesh_seq._create_indicators() # this should create indicators as a non-private funciton
        # TODO: as Joe about this as it seems inefficient - why not just add inicators to the 
            # parent class on init?
    
        if "hessian" in self.params["adaptor_method"] or "uniform" in self.params["adaptor_method"]:
            
            # final_solution =self.mesh_seq.__class__.mro()[3].fixed_point_iteration( 
            #         self.mesh_seq, 
            #         self.adaptor.adaptor,
            #         parameters = self.params 
            # )


            # print(f"\n\n FINAL SOLUTION {final_solution}  -  {final_solution.extract()['c']['forward'][-1][-1]} {self.mesh_seq.solutions}")
            # calls the fixed point iteration from the MeshSeq parent class directly
            # TODO: is this the best solution for this? Especially now that the 
                # outputs of solution and/or indicator are saved on the MeshSeq object directly?
            final_solution = gol_adj.MeshSeq.fixed_point_iteration( 
                self.mesh_seq, 
                self.adaptor.adaptor,
                parameters = self.params   
            )
            print(f"\n\n post fpi type solution {final_solution}")
            # self.adaptor.solutions.append(final_solution)

        # if ML method need which needs adjoint
        elif  self.params["indicator_method"] in ["gnn","mlp","gnn_noadj",]:
            final_solution =gol_adj.AdjointMeshSeq.fixed_point_iteration( 
                self.mesh_seq, 
                self.adaptor.adaptor,
                parameters = self.params 
            )
        else:
            # print(f'\n\n SIM {self.params["enrichment_kwargs"]}\n\n')
            final_solution =self.mesh_seq.fixed_point_iteration( 
                self.adaptor.adaptor,
                enrichment_kwargs=self.params["enrichment_kwargs"],
                # adaptor_kwargs=self.params["adaptor_kwargs"]
                parameters = self.params  
            )
            # print("in simulation block for indicate reros")
            # solu = self.mesh_seq.indicate_errors(enrichment_kwargs=self.params["enrichment_kwargs"] )
            # self.adaptor.solutions.append(final_solution[0])

        # timer end
        duration += perf_counter()
        # log time:
        
        if self.params["adaptor_method"] == 'uniform':
            logging.info(f'FPITER, {len(self.adaptor.mesh_stats)}, {self.params["adaptor_method"]}')
            logging.info(f'FTIMING, fp, {duration:.6f}, {self.params["adaptor_method"]}')
            logging.info(f'FQOI, fp, {self.adaptor.qois[-1]}, {self.params["adaptor_method"]}')
            logging.info(f'FDOF, fp, {self.adaptor.mesh_seq[0].num_vertices()}, {self.params["adaptor_method"]}')
        else:
            logging.info(f'FPITER, {len(self.adaptor.mesh_stats)}, {self.params["adaptor_method"].split("_")[0]}, {self.params["adaptor_method"].split("_")[1]}')
            logging.info(f'FTIMING, fp, {duration:.6f}, {self.params["adaptor_method"].split("_")[0]}, {self.params["adaptor_method"].split("_")[1]}')
            logging.info(f'FQOI, fp, {self.adaptor.qois[-1]}, {self.params["adaptor_method"].split("_")[0]}, {self.params["adaptor_method"].split("_")[1]}')
            logging.info(f'FDOF, fp, {self.adaptor.mesh_seq[0].num_vertices()}, {self.params["adaptor_method"].split("_")[0]}, {self.params["adaptor_method"].split("_")[1]}')
        print(f' fixed point iterator total time taken: {duration:.6f} seconds, {len(self.adaptor.mesh_stats)} iterations')

        # QC plot for fixed point loop statistics:
        self.adaptor.plot_mesh_convergence_stats(6,2)

        # Exports - not sure if needed??
        # if the fpi iteration count is larger than adapt iteration count, 
        # convergence due to QOI or Indicatior so the final solution was not exported
        print(f" \n\n COUNT POST FPI: fpi count {self.adaptor.mesh_seq.fp_iteration}, adaptor count {self.adaptor.adapt_iteration}")
        if self.adaptor.mesh_seq.fp_iteration == self.adaptor.adapt_iteration:
            print(f"\n final solution also output")
            self.adaptor.adapt_outputs(self.mesh_seq, self.params["adaptor_method"])

        # self.adaptor.plot_adapt_meshes() # ej321 - relies on capturing all simulations in list at runtime

        # reset working directory to default
        os.chdir(f"{self.rootfolder}")

        logging.shutdown()

        

if __name__ == "__main__":
    pass