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
from workflow.timer import Timer
import json

import shutil
import argparse

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
            
    if filepath is None:
        filepath = rootfolder

        if not os.path.isdir(os.path.join(filepath,foldername)):
            # TODO: add MPI for running on HPC?
            os.makedirs(os.path.join(filepath,foldername)) 
            filepath = os.path.join(filepath,foldername)
            print(f'simulation folder created: {filepath}') 
            return filepath
        else:
            filepath = os.path.join(filepath,foldername)
            print(f'simulation using folder: {filepath}')
            return filepath
        
    if os.path.isdir(filepath):
        print(f'simulation using folder: {filepath}')
        return filepath

     



class Simulation():
    
    def __init__(self, Model, rootfolder=None, filepath=None, parameters={}):
        self.rootfolder = rootfolder if rootfolder else os.getcwd()
        self.params = self.create_params(Model, parameters)
        self.filepath = create_folder(self.rootfolder, str(Model.__name__), filepath)
        self.Model = Model
        self.local_filepath = None

    def create_params(self, Model, parameters):
        """
        combine all parameters into one dictionary and export to json
        note: done to pass all parameters to Model in blind init given setup of Goalie
        
        adding feature from Joe's code parameter class to pass keys as methods
        TODO: figure out a better place for this
        """
        params=AttrDict() # TODO: set some defaults for all parameters passed      
        params.update(Model.get_default_parameters())
        params.update(parameters)
        # Try ARG PARSE HERE?
        parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
        )
        def coords(s):
            """
            # https://stackoverflow.com/questions/9978880/python-argument-parser-list-of-list-or-tuple-of-tuples

            """
            try:
                situp = []
                for si in s.split('.'):
                    situp.append(tuple(map(int, si.split(','))))
                print(situp)
                return situp
            except:
                raise argparse.ArgumentTypeError("Coordinates must be\
                given divided by commas and dot e.g.: 'x,y.k,l,m'")

        def namespace_to_dict(namespace):
            """
            to convert nested Namespace to dictionary format recursively
            from stackoverflow
            https://stackoverflow.com/questions/71845452/convert-argparse-namespace-to-dict-recursively
            """
            return {
                k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
                for k, v in vars(namespace).items()
            }
        

        # TODO: this will have issues with nested dictionaries, need check for
        for key, value in params.items():
            print(key, value, type(value))
            if key == "coordinates":
                parser.add_argument(f"--{key}", type=coords)
            else:
                parser.add_argument(f"--{key}", type=type(value))
        # parse user defined folder name if specified
        # TODO: set up to concatenate names if multiple given, but
        #       may want to change this structure later
        parser.add_argument(f"--casefolder", type=str, action='append')
        
        args = parser.parse_args()
        print(f'args passed from CLI: {namespace_to_dict(args)}')
        print(f'casefolder all : {args.casefolder}')

        # define as new subfolder
        subfolder = "/"
        for case in args.casefolder:
            subfolder = subfolder + f"{case}_"
        # strip last "_"
        subfolder = subfolder[:-1]

        print(f' casefolder specified as: {subfolder}')
        self.rootfolder = self.rootfolder + subfolder

        #update parameters with argparsed
        params.update(namespace_to_dict(args))
        print(params)

        return params



    def setup_logging_file(self, filepath):
        # need to remove logging handler explicity to switch filepath if one has
        # previously initialised
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO, 
            filename=os.path.join(filepath,"timesout.txt"),
            format='%(asctime)s - %(levelname)s - %(message)s'
            )
    
    def set_outfolder(self, suffix=""):
        # TODO: fix to only create file if actually output

        self.local_filepath = f"{self.filepath}/{suffix}"
        if  not os.path.isdir(self.local_filepath):
            os.makedirs(self.local_filepath) 
        
        # temporarily set this as the current working directory
        os.chdir(self.local_filepath)

        # QC:
        print(f'current working directory changed to: {self.local_filepath}')

    def output_params_json(self):
        """
        dump parameter dictionary to json file in local file path
        """
        # TODO: this could be a static method as self not really needed
        with open(f'{self.local_filepath}/input_parameters.json', 'w') as fp:
            json.dump(self.params, fp)
        print(f"writing parameters to json: {self.local_filepath}")




    @PETSc.Log.EventDecorator("SIMULATION")
    @Timer(logger = logging.info, text="full simulation, time taken: {:.6f}")
    def run_simulation(self):

        adaptor_method = self.params["adaptor_method"]
        # set output folder with current date
        self.set_outfolder(f'{adaptor_method}_{datetime.now().strftime("%H%M%S")}')
        
        # setup logging
        self.setup_logging_file(self.local_filepath)
        logging.info(f'Output Folder: {os.getcwd()}')
        logging.info(f'Input parameters: {self.params}')

        # run FPI
        # output parameters for init of mesh_seq
        self.output_params_json()

        # # call instance of mesh_seq
        mesh_seq = self.Model(
            self.Model.get_default_partition(**self.params),
            self.Model.get_default_meshes(self.filepath,**self.params),
            qoi_type=self.params["qoi_type"]
            )
        
        # # call instance of adaptor
        adaptor = AdaptorSelection(adaptor_method, **self.params)

        with Timer(logger = logging.info, text="fixed point iteration loop, time taken: {:.6f}"):

            if "hessian" in self.params["adaptor_method"] or "uniform" in self.params["adaptor_method"]:
                
                # calls the fixed point iteration from the MeshSeq parent class directly
                # TODO: is this the best solution for this? Especially now that the 
                    # outputs of solution and/or indicator are saved on the MeshSeq object directly?
                gol_adj.MeshSeq.fixed_point_iteration( 
                    mesh_seq, 
                    adaptor.adaptor,
                    parameters = self.params   
                )

            # if ML method need which needs adjoint
            elif  self.params["indicator_method"] in ["gnn","mlp","gnn_noadj",]:
                print('\n\n\n\n\t\t\tAdjoint solver')
                gol_adj.AdjointMeshSeq.fixed_point_iteration( 
                    mesh_seq, 
                    adaptor.adaptor,
                    parameters = self.params 
                )

            else:
                mesh_seq.fixed_point_iteration( 
                    adaptor.adaptor,
                    enrichment_kwargs=self.params["enrichment_kwargs"],
                    # adaptor_kwargs=self.params["adaptor_kwargs"]
                    parameters = self.params  
                )

        # timer end
        # duration += perf_counter()
        logging.info(f'fixed point iterator total iterations: {len(adaptor.mesh_stats)}')
        # logging.info(f'fixed point iterator total time: {duration:.6f}')
        # QC plot for fixed point loop statistics:
        adaptor.plot_mesh_convergence_stats(mesh_seq,plot_len =6,subplot_ht=2)

        # reset working directory to default
        os.chdir(f"{self.rootfolder}")

        # shut off logging
        logging.shutdown()



    def run_fp_loop(self):

        # set output folder with current date
        self.set_outfolder(f'{datetime.now().strftime("%H%M%S")}')
        self.setup_logging_file(self.local_filepath)
        logging.info(f'\n################################################')
        logging.info(f'\n\tFIXED_POINT_ITERATION_FOLDER: {os.getcwd()}')
        logging.info(f'\n parameters: {self.params}')
        
        # timer start:
        duration = -perf_counter()

        # TODO: this as it seems inefficient - why not just add indicators to the parent class on init?
        self.mesh_seq._create_indicators() # this should create indicators as a non-private function
        

        if "hessian" in self.params["adaptor_method"] or "uniform" in self.params["adaptor_method"]:
            
            # calls the fixed point iteration from the MeshSeq parent class directly
            # TODO: is this the best solution for this? Especially now that the 
                # outputs of solution and/or indicator are saved on the MeshSeq object directly?
            gol_adj.MeshSeq.fixed_point_iteration( 
                self.mesh_seq, 
                self.adaptor.adaptor,
                parameters = self.params   
            )

            # alternative:
            # final_solution =self.mesh_seq.__class__.mro()[3].fixed_point_iteration( 
            #         self.mesh_seq, 
            #         self.adaptor.adaptor,
            #         parameters = self.params 
            # )

        # if ML method need which needs adjoint
        elif  self.params["indicator_method"] in ["gnn","mlp","gnn_noadj",]:
            print('Adjoint solver')
            gol_adj.AdjointMeshSeq.fixed_point_iteration( 
                self.mesh_seq, 
                self.adaptor.adaptor,
                parameters = self.params 
            )

            # alternative:
            # self.mesh_seq.__class__.mro()[2].fixed_point_iteration( 
            #         self.mesh_seq, 
            #         self.adaptor.adaptor,
            #         parameters = self.params 
            # )

        else:
            self.mesh_seq.fixed_point_iteration( 
                self.adaptor.adaptor,
                enrichment_kwargs=self.params["enrichment_kwargs"],
                # adaptor_kwargs=self.params["adaptor_kwargs"]
                parameters = self.params  
            )

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

        # QC - check alignment of counters
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