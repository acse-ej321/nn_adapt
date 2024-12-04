
import firedrake as fd
import numpy as np

import goalie_adjoint as gol_adj # new sept23
# from workflow.utility import *
from workflow.features import * # ej321 - for feature extraction

import thetis as thetis
import pygmsh
import gmsh

from collections import OrderedDict
from time import perf_counter
import logging
import json

from firedrake.__future__ import interpolate # ej321 - temp for firedrake update

import os


def get_parameters_from_json(filepath):
    """
    load parameter file from json file - this in not ideal
    but a work around to pass parameters not allowed through the
    Goalie API currently.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as fp:
            params = json.load(fp)
        print(f"load parameters from file: {filepath}")
    else: 
        params = TurbineMeshSeq.get_default_parameters()
        print(f"load parameters from defaults: {filepath}")
    return params


class TurbineMeshSeq(gol_adj.GoalOrientedMeshSeq):
    
    def __init__(self, *args, **kwargs):
        self.thetis_manager={}
        self.filepath = kwargs.get("local_filepath",
                                 os.getcwd())
        # TODO: check if this still utilized
        self.model_features={} # add for feature extraction ala E2N - model specific
        super().__init__(*args, **kwargs)
        # self.params =kwargs.get('parameters',
        #                          self.get_default_parameters())
        self.params =kwargs.get('parameters',
                                 get_parameters_from_json(f"{self.filepath}/input_parameters.json"))
    @staticmethod
    def get_default_parameters():
        return {
                "enrichment_kwargs":{
                    "enrichment_method": "h",
                    # "num_enrichments": 1
                },
                # field kwargs
                "field":"solution_2d",
                "fields":["solution_2d"],
                "num_meshes":1,
                # qoi kwargs
                "qoi_name":"power output",
                "qoi_unit":"MW",
                # simulation kwargs
                "output_directory":'outputs_nn_adapt', # for thetis
                "viscosity_coefficient":0.5,
                "depth": 40.0,
                "drag_coefficient":0.0025,
                "inflow_speed": 5.0,
                "density":1030.0 * 1.0e-06,
                "turbine_diameter":18.0,
                "turbine_width": None,
                "turbine_coords": [],
                "thrust_coefficient": 0.8,
                "correct_thrust": True, # ej321 - test out; default is True
                "solver_parameters": {
                    "mat_type": "aij",
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1.0e-08,
                    "snes_max_it": 100,
                    "snes_monitor": None, # added
                    "ksp_type": "preonly",
                    "ksp_converged_reason": None, # added
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
                "coordinates":[(456,250), (744,250)],
                "discrete":False, # ej321 - Joe has this as False
                # fpi kwargs
                "qoi_type": 'steady',
                "miniter":3,
                "maxiter" : 35,
                "base": 200,
                "target":4000,
                "element_rtol": 0.001, 
                "qoi_rtol": 0.001, 
                "estimator_rtol":0.001,
                "drop_out_converged": False,
                "convergence_criteria":"any", # options are "all" or "any"
                
                # adaptation kwargs
                "indicator_method": "gnn", # added as flag for fp iteration, options "gnn"
                "adaptor_method":"steady_anisotropic",
                "fix_boundary": False,
                "area_labels": [],
                "fix_area": False,
                "indicator_method" : None,

                #metric kwargs
                "h_min":1.0e-8,
                "h_max":500.0,
                'a_max':1.0e5, # used:
                'interpolant': 'Clement', # used:
                'average': True, # used:
                'retall': True, # used:
                'dm_plex_metric_p': np.inf,
            }

    @staticmethod
    def get_default_meshes(filepath, **kwargs):
   
        # Get local path if mesh exists
        # linux:
        local_path= 'data0'
        local_folder = 'nn_adapt/models/inputs'

        # pc: 
        # local_path = 'home/phd01'
        # local_folder = 'adaptwkflw/models/inputs'

        # file_name = 'aligned_dec23.msh'
        # meshpath = f"/{local_path}/{local_folder}/{file_name}"
        # print(os.getcwd())

        # mesh = TurbineMeshSeq.create_turbine_mesh()

        # create new mesh
        meshpath = TurbineMeshSeq.create_turbine_mesh(filepath)


        def create_default_meshes(num_meshes,**kwargs):

            meshes = []
            #this works to generate independent meshes per partition
            if num_meshes > 1:
                for m in range(num_meshes):
                   meshes.append(fd.Mesh(meshpath,name=f'mesh_0_{m}'))
            else:
                meshes.append(fd.Mesh(meshpath, name=f'mesh_0_0'))
            
            # TODO: save out copy of meshes in input folder
            # QC
            print(os.getcwd())
            print(f'\n\n\ndefault meshes: {meshes}')

            return meshes
        
        return create_default_meshes(**kwargs)


    @staticmethod
    def get_default_partition(**kwargs):
        # TODO: get adjusted values back into parameters dictionary for documentation and/or
        # into the log file
        def create_default_partition(fields, **kwargs):

            return gol_adj.TimeInstant(fields)

        return create_default_partition(**kwargs)


    def get_model_features(self, mesh):
        # addtitional parameters for E2N paper
        self.model_features["global_inflow_speed"]= self.params["inflow_speed"]
        # adding extract first element of tuple
        # TODO - figure out why exporting as a tuple here
        self.model_features["physics_drag"] =  extract_array(self.drag(mesh)),
        self.model_features["physics_viscosity"]= extract_array(self.viscosity(mesh), project=True),
        self.model_features["physics_bathymetry"]= extract_array(self.bathymetry(mesh), project=True),

        # print(f'\n\n GET MODEL FEATURES: {self.model_features["physics_bathymetry"]}')

    # SPECIAL FOR THETIS -------------------------------------------------------VVVVVVVVVVV
    
    @staticmethod
    def add_turbine(model, coordinates, D, dx_inner, z=0):
        """
        returns a mesh rectangle objects
        
        model: gmsh model object
        coordinates: list of coordinates[(x0,y0),...]
        D: diameter of rectangle
        dx_inner: target mesh size
        z: 3rd dimension, optional
        
        requires gmsh and pygmesh
        
        """
        rec_list=[]
        for xt0,yt0 in coordinates:
            site_x_start = xt0-D/2
            site_x_end = xt0+D/2

            site_y_start = yt0-D/2
            site_y_end = yt0+D/2
            rec = model.add_rectangle(site_x_start, site_x_end, site_y_start, site_y_end, z, 
                            mesh_size=dx_inner,
                            holes=None, make_surface=True)
        #     rec_surf=model.add_plane_surface(rec.curve_loop)
            rec_list.append(rec)
        return rec_list

    @staticmethod
    def create_turbine_mesh(meshpath, filename="inital_turbine_mesh.msh", m=0, L=1200.,W=500.,D=18., z=0., 
                            dx_inner=20.,dx_outer=20., n_min=1, n_max=8):
        
        dim=2 #2D
        algorithm=8 #same as Joes E2N paper
        
        # Domain and turbine specification
        x_min= 50
        x_max= L-100
        y_min=50
        y_max=W-100
        #coordinates = rand_2d_coords(x_min,x_max,y_min,y_max,n_min=3,n_max=4)
        # TODO: allow non default parameters
        params = TurbineMeshSeq.get_default_parameters()
        coordinates = params["coordinates"]

        #QC
        print(f'create turbine coordinates {coordinates}')
        # Initialize empty geometry using the build in kernel in GMSH
        # geometry = pygmsh.geo.Geometry() #

        # Fetch model we would like to add data to
        # model = geometry.__enter__()

        with pygmsh.geo.Geometry() as model:
            
            # Get a list of the turbine mesh rectangle objects
            rec_list=TurbineMeshSeq.add_turbine(model,coordinates,D, dx_inner)
        
            # Create outer mesh and add turbines in as 'holes'
            rec_outer = model.add_rectangle(0, L, 0, W, z, 
                                    mesh_size=dx_outer,
                                holes=[rec.curve_loop for rec in rec_list], make_surface=True)
        
            # rec_surf=model.add_plane_surface(rec.curve_loop) # may not need
        
            # Call gmsh kernel before add physical entities
            model.synchronize()
        
            # Add Physical Entites - boundary labels
            model.add_physical(rec_outer.lines[3], "Inflow")  #Left Boundary
            model.add_physical(rec_outer.lines[1], "Outflow")  #Right Boundary
            model.add_physical([rec_outer.lines[0], rec_outer.lines[2]], "Walls")  #Sides
            model.add_physical(rec_outer.surface, "Volume")  # Outside loop
        
            # Add label for all turbines
            [model.add_physical(rec.surface,f'Turbine_{i}') for i,rec in enumerate(rec_list)]


            # Works
            # geometry.generate_mesh(dim=dim,algorithm=algorithm,verbose=True)
            model.generate_mesh(dim=dim,algorithm=algorithm,verbose=True)
            # gmsh.initialize()
            meshfile = os.path.join(meshpath, filename)
            gmsh.write(meshfile) #needs extension?
            # gmsh.clear()
            # geometry.__exit__()
        
        # return the coordinates of the turbine locations
        # return fd.Mesh(filename, name=f"mesh_0_{m}")
        return meshfile

    def get_flowsolver2d(self, mesh, initial_condition, **kwargs):
        """
        :arg mesh: the mesh to define the solver on
        :arg ic: the initial condition
        """
        bathymetry = self.bathymetry(mesh)

        # Create solver object
        thetis_solver = thetis.solver2d.FlowSolver2d(mesh, bathymetry)

        # ej321 - turns off initial callback throwing off integrated power calculation
        thetis_solver.export_initial_state = False 
        options = thetis_solver.options

        # thetis output checks for QC
        options.output_directory = self.params['output_directory']
        options.fields_to_export = ['elev_2d', 'uv_2d','Bathymetry']

        # time stepper options  ej321 - add for understanding
        options.timestep = 20.0
        options.simulation_export_time = 20.0
        options.simulation_end_time = 18.0
        options.swe_timestepper_type = "SteadyState"
        options.swe_timestepper_options.ad_block_tag="solution_2d" # ej321 - for adjoint
        # options.swe_timestepper_options.ad_block_tag="solution" # for adjoint
        options.swe_timestepper_options.solver_parameters = self.params['solver_parameters']
        
        # parameter options ej321 - add for understanding
        options.element_family = "dg-cg"
        options.horizontal_viscosity = self.viscosity(mesh)
        options.quadratic_drag_coefficient = fd.Constant(self.params['drag_coefficient'])
        # other
        options.use_grad_div_viscosity_term = False
        options.use_lax_friedrichs_velocity = True
        options.lax_friedrichs_velocity_scaling_factor = fd.Constant(1.0)
        options.use_grad_depth_viscosity_term = False

        options.no_exports = False
        options.update(kwargs)
        # self._thetis_solver.create_equations() # ej321 moving to after all options set

        # Apply boundary conditions
        # P1v_2d = self._thetis_solver.function_spaces.P1v_2d # ej321 - commented out
        P1v_2d = thetis.get_functionspace(mesh, "DG", 1, vector=True) # ej321 - added
        u_inflow = fd.assemble(interpolate(self.u_inflow(mesh), P1v_2d))

        # check for flow direction and adjust boundaries
        # TODO: seperate into get_bc function with options
        if self.params["inflow_speed"]<0:
            inflow_bnd = 2 # from right
            outflow_bnd = 1 # to left
        else:
            inflow_bnd = 1 # from left
            outflow_bnd = 2 # to right
        #flipped boundaries
        thetis_solver.bnd_functions["shallow_water"] = {
            inflow_bnd: {"uv": u_inflow},  # inflow, left
            outflow_bnd: {"elev": fd.Constant(0.0)},  # outflow, right
            3: {"un": fd.Constant(0.0)},  # free-slip, sides
            4: {"uv": fd.Constant(fd.as_vector([0.0, 0.0]))},  # no-slip ej321 - will this work, is noslip adjoint with constants fixed?
            5: {"elev": fd.Constant(0.0), "un": fd.Constant(0.0)},  # weakly reflective, turbine 1
            6: {"elev": fd.Constant(0.0), "un": fd.Constant(0.0)}  # weakly reflective, turbine 2
        }

        print(f"for mesh {mesh} - {thetis_solver.bnd_functions['shallow_water']}")

        # Create tidal farm
        # ej321 - using continuous farm case as discrete?
        # options.tidal_turbine_farms = self.farm(mesh) # ej321 for TidalTurbineFarmOptions

        # options.tidal_turbine_farms["everywhere"] = [self.farm(mesh)]
        options.discrete_tidal_turbine_farms["everywhere"] = [self.farm(mesh)] # ej321 - for discrete
        # options.discrete_tidal_turbine_farms = self.farm(mesh)
        thetis_solver.create_equations() # ej321 moving to after all options set


        # Apply initial guess
        # u_init, eta_init = ic.subfunctions
        vel_init, elev_init = initial_condition.subfunctions
        thetis_solver.assign_initial_conditions(uv=vel_init, elev=elev_init)

        # Apply initial guess
        #u_init, eta_init = ic.split() #ej321
        # u_init, eta_init = ic.subfunctions #ej321 - added field for goalie consisitency
        # thetis_solver.assign_initial_conditions(uv=u_init, elev=eta_init)

        cb = thetis.turbines.TurbineFunctionalCallback(thetis_solver)
        
        thetis_solver.add_callback(cb)

        turbine_density = thetis.Function(thetis_solver.function_spaces.P1_2d, name='turbine_density')
        

        # ej321 - check if this call to File is throwing warnings
        # turbine_density.interpolate(thetis_solver.tidal_farms[0].turbine_density)
        # thetis.File('turbine_density.pvd').write(turbine_density)
        return thetis_solver

    def farm(self, mesh):
        """
        Construct a dictionary of :class:`TidalTurbineFarmOptions`
        objects based on the current `mesh`.
        """
        turbine_diameter = self.params["turbine_diameter"]

        coordinates =  self.params["coordinates"]
        # print(f'\n farm coordinates: {coordinates}')

        #----- # ej321 - commented out, as using 
        # farm_options = thetis.TidalTurbineFarmOptions()

        # custom turbine density
        # farm_options.turbine_density = self.turbine_density(mesh) 
        # farm_options.turbine_options.diameter = turbine_diameter
        # farm_options.turbine_options.thrust_coefficient= self.params["thrust_coefficient"] 

        # # return {farm_id: [farm_options] for farm_id in [5,6]} # ej321 - explicit for 2 turbine case
        # return {farm_id: [farm_options] for farm_id in [2,3]} # ej321 - explicit for 2 turbine case
        # return farm_options
        #---


        # --- # ej321 - update for discrete turbines
        farm_options = thetis.DiscreteTidalTurbineFarmOptions() # ej321 - update for discrete turbines
        farm_options.turbine_type = 'constant'# ej321 - update for discrete turbines
        
        farm_options.turbine_options.diameter = turbine_diameter
        farm_options.turbine_options.thrust_coefficient= self.params["thrust_coefficient"] # don't apply joes correction factor
        farm_options.quadrature_degree = 3 # ej321 - set to see if this is the convergence difference
        # # added specifically for discrete case WORKED
        # farm_options.upwind_correction = False # higher SNES tolerance required when using upwind correction
        farm_options.turbine_coordinates = coordinates

        # print(farm_options.turbine_coordinates)
        # #----
        # return {farm_id: [farm_options] for farm_id in [2,3]} 
        return farm_options


    @property
    def num_turbines(self):
        """
        Count the number of turbines based on the number
        of coordinates.
        """
        return len(self.turbine_coords)

    @property
    def turbine_ids(self):
        """
        Generate the list of turbine IDs, i.e. cell tags used
        in the gmsh geometry file.
        """
        if self.params["discrete"]:
            return list(2 + np.arange(self.num_turbines, dtype=np.int32))
        else:
            return ["everywhere"]

    @property
    def footprint_area(self):
        """
        Calculate the area of the turbine footprint in the horizontal.
        """
        d = self.params["turbine_diameter"]
        w = self.params["turbine_width"] or d
        return d * w

    @property
    def swept_area(self):
        """
        Calculate the area swept by the turbine in the vertical.
        """
        return fd.pi * (0.5 * self.params["turbine_diameter"]) ** 2

    @property
    def cross_sectional_area(self):
        """
        Calculate the cross-sectional area of the turbine footprint
        in the vertical.
        """
        return self.params["depth"] * self.params["turbine_diameter"]

    @property
    def corrected_thrust_coefficient(self):
        """
        Correct the thrust coefficient to account for the
        fact that we use the velocity at the turbine, rather
        than an upstream veloicity.

        See [Kramer and Piggott 2016] for details.
        """
        Ct = self.params["thrust_coefficient"]
        correct_thrust = self.params["correct_thrust"]
        depth = self.params["depth"]
        turbine_diameter = self.params["turbine_diameter"]

        cross_sectional_area = depth * turbine_diameter
        swept_area = fd.pi * (0.5 * turbine_diameter) ** 2
        if not correct_thrust:
            return Ct
        
        corr = 4.0 / (1.0 + fd.sqrt(1.0 - Ct * swept_area / cross_sectional_area)) ** 2
        return Ct * corr

    def bathymetry(self, mesh):
        """
        Compute the bathymetry field on the current `mesh`.
        """
        # NOTE: We assume a constant bathymetry field
        depth =self.params['depth']
        P0_2d = thetis.get_functionspace(mesh, "DG", 0)
        return fd.Function(P0_2d).assign(depth)

    # def u_inflow(self, mesh):
    def u_inflow(self, mesh):
        """
        Compute the inflow velocity based on the current `mesh`.
        """
        # NOTE: We assume a constant inflow
        inflow_speed=self.params["inflow_speed"]
        print(f"\n\n\tin u_inflow {inflow_speed}")
        return fd.as_vector([inflow_speed, 0])

    # def ic(self, mesh):
    #     """
    #     Initial condition.
    #     """
    #     return self.u_inflow(mesh)

    def turbine_density(self, mesh):
        """
        Compute the turbine density function on the current `mesh`.
        """

        turbine_diameter = self.params["turbine_diameter"]

        if self.params["discrete"]:
            # print(1.0 / self.footprint_area)
            R = fd.FunctionSpace(mesh,"R",0)
            return fd.Function(R).assign(1.0 / self.footprint_area)

            # return fd.Constant(1.0 / self.footprint_area, domain=mesh)
            # return 1.0 / self.footprint_area

        x, y = fd.SpatialCoordinate(mesh)
        r2 = turbine_diameter / 2

        r1 = r2 if self.params["turbine_width"] is None else self.params["turbine_width"] / 2

        def bump(x0, y0, scale=1.0):
            qx = ((x - x0) / r1) ** 2
            qy = ((y - y0) / r2) ** 2
            cond = fd.And(qx < 1, qy < 1)
            b = fd.exp(1 - 1 / (1 - qx)) * fd.exp(1 - 1 / (1 - qy))
            return fd.conditional(cond, fd.Constant(scale) * b, 0)

        bumps = 0


        for xy in self.params["coordinates"]:
            
            bumps += bump(*xy, scale=1 / fd.assemble(bump(*xy) * fd.dx))
            # print(f'\n\n EJ DISCRETE TURBINE ADDED {xy} turbine density {bumps}') # ej321
        
        return bumps


    def turbine_drag(self, mesh):
        """
        Compute the contribution to the drag coefficient due to the
        tidal turbine parametrisation on the current `mesh`.
        """
        P0_2d = thetis.get_functionspace(mesh, "DG", 0)
        p0test = fd.TestFunction(P0_2d)
        Ct = self.corrected_thrust_coefficient
        At = self.swept_area
        Cd = 0.5 * Ct * At * self.turbine_density(mesh)
        print(f'JOE TURBINE DRAG')
        return sum([p0test * Cd * fd.dx(tag, domain=mesh) for tag in self.turbine_ids])

    def drag(self, mesh, background=False):
        r"""
        Create a :math:`\mathbb P0` field for the drag on the current
        `mesh`.

        :kwarg background: should we consider the background drag
            alone, or should the turbine drag be included?
        """
        P0_2d = thetis.get_functionspace(mesh, "DG", 0)
        ret = fd.Function(P0_2d)

        # Background drag
        Cb = self.params["drag_coefficient"]
        if background:
            return ret.assign(Cb)
        p0test = fd.TestFunction(P0_2d)
        expr = p0test * Cb * fd.dx(domain=mesh)

        # Turbine drag
        fd.assemble(expr + self.turbine_drag(mesh), tensor=ret)
        return ret

    def viscosity(self, mesh):
        r"""
        Create a :math:`\mathbb P0` field for the viscosity coefficient
        on the current `mesh`.
        """
        # NOTE: We assume a constant viscosity coefficient
        viscosity_coefficient = self.params['viscosity_coefficient']
        P0_2d = thetis.get_functionspace(mesh, "DG", 0)
        return fd.Function(P0_2d).assign(viscosity_coefficient)

    @property
    def solution(self):
        return self._thetis_solver.fields.solution_2d

    def manage_thetis_object(self, mesh, ic = None):

        # if dictionary doe not exist, create it
        # this would indicate the firt time it is called
        # print(self.thetis_manager)
        if not bool(self.thetis_manager):
            print("empty dictionary")

        # if the dictionary exists, check for the current mesh object
        # else:
        if mesh in self.thetis_manager.keys():
            print(f"mesh in manager")
            return self.thetis_manager[mesh]
        
        # if the mesh is not in the dictionary
        else:
            # add the mesh to the dictionary
            print(f"add mesh to manager")
            # presumably the first run such that we fix the function space
            # for the problem to avoid recursive loop
            # P1v_2d = thetis.get_functionspace(mesh, "DG", 1, vector=True)
            # P2_2d = thetis.get_functionspace(mesh, "CG", 2)
            # _fs = P1v_2d * P2_2d

            # get the goalie index for the current mesh:
            mesh_index = self.meshes.index(mesh)

            # asign thetis flow solver object to dicitonary
            _fs = self._initial_fs(mesh_index)
            solv=self.get_flowsolver2d(mesh, self._intial_condition(_fs))

            print(f'solv {solv}')

            self.thetis_manager[mesh] = solv
            # print(self.thetis_manager)
    
            # retrun the thetis solver
            return self.thetis_manager[mesh] 


    def _initial_fs(self, index):
        
        mesh = self[index]
        P1v_2d = thetis.get_functionspace(mesh, "DG", 1, vector=True)
        P1_2d = thetis.get_functionspace(mesh, "CG", 1)
        _ifs = P1v_2d * P1_2d

        print(f'\n in intial fs, {index} - mesh {mesh}')

        return _ifs


    def _intial_condition(self, _ifs):

        print(f"\n\n initial condition - mesh {self}")

        # define initial condition function from initial function space
        _ic =   fd.Function(_ifs)
        vel_init, elev_init = _ic.subfunctions # fd.split(q) # avoid subfunctions?
        #TODO: mesh passed but never utilized
        vel_init.interpolate(self.u_inflow(_ifs.mesh()))

        print(f'in initial conditions {vel_init.dat.data[:]}')
        print(self.params["inflow_speed"])

        return _ic


    # SPECIAL FOR THETIS -------------------------------------------------------^^^^^^^^^^

    # @staticmethod
    def get_function_spaces(self,mesh):
        """
        Construct the (mixed) finite element space used for the
        prognostic solution.
        """
        field=self.params["field"]  

        print(f"get function spaces, mesh {mesh}")
        thetis_obj=self.manage_thetis_object(mesh)

        # return {"solution_2d":self._thetis_solver.function_spaces.V_2d}
        return {field: thetis_obj.fields.solution_2d.function_space()}


    # @staticmethod
    # def source(mesh, x0, y0, r, scale):
    #     x, y = fd.SpatialCoordinate(mesh)
    #     return scale * fd.exp(-((x - x0) ** 2 + (y - y0) ** 2) / r**2)

    def get_initial_condition(self):
        """
        Compute an initial condition based on the inflow velocity
        and zero free surface elevation.
        """
        field=self.params["field"]  
        thetis_obj=self.manage_thetis_object(self[0])
        _fs = thetis_obj.fields.solution_2d.function_space()
        # P1v_2d = thetis.get_functionspace(self[0], "DG", 1, vector=True)
        # P2_2d = thetis.get_functionspace(self[0], "CG", 2)
        # _fs = P1v_2d * P2_2d
        # rebuild initial conditions
        q = self._intial_condition(_fs)

        return {field: q}

    # def get_form(self, field="solution_2d"):
    
    #     def form(index):
    #     # def form(index, sols):
    #         thetis_obj=self.manage_thetis_object(self[index])

    #         #The weak form of the shallow water equations.
    #         return {"solution_2d":thetis_obj.timestepper.F}
        
    #     return form

    # UDPATE
    # def get_bcs(self):
    #     def bcs(index,field="solution_2d"):
    #         # Apply boundary conditions
    #         # P1v_2d = self._thetis_solver.function_spaces.P1v_2d # ej321 - commented out
    #         P1v_2d = thetis.get_functionspace(mesh, "DG", 1, vector=True) # ej321 - added
    #         u_inflow = fd.assemble(interpolate(parameters.u_inflow(mesh), P1v_2d))
    #         self._thetis_solver.bnd_functions["shallow_water"] = {
    #             1: {"uv": u_inflow},  # inflow, left
    #             2: {"elev": fd.Constant(0.0)},  # outflow, right
    #             3: {"un": fd.Constant(0.0)},  # free-slip, sides
    #             4: {"uv": fd.Constant(fd.as_vector([0.0, 0.0]))},  # no-slip ej321 - will this work, is noslip adjoint with constants fixed?
    #             5: {"elev": fd.Constant(0.0), "un": fd.Constant(0.0)}  # weakly reflective
    #         }

    #     return bcs


    def get_solver(self):

        # def solver(index, ic):
        def solver(index):
            # print(f'\n\nEJ CHECK solver called')

            # timer start:
            duration = -perf_counter() # ej321 - TIMING

            field=self.params["field"]
            # thetis_obj=self.get_flowsolver2d(self[index], ic[field])

            thetis_obj=self.manage_thetis_object(self[index])
            thetis_obj.simulation_time=0.

            # Assign initial condition
            # _sol, _sol_old = self.fields[field] # current, last/ic
            _sol = self.fields[field] # current, last/ic
            # c.assign(c_)
            # thetis_obj.fields.solution_2d.assign(ic[field])
            # thetis_obj.fields.solution_2d.assign(_sol_old)

            # Communicate variational form to mesh_seq
            self.read_forms({field:thetis_obj.timestepper.F})
            

            thetis_obj.fields.solution_2d.assign(_sol)
            

            print(f"\n BEFORE ITERATE {thetis_obj.callbacks['export']['turbine'].integrated_power[0]}\n\n")
            


            iterate = thetis_obj.get_iterate()
            # thetis_obj.iterate_ej321()

            iterate()

            _sol.assign(thetis_obj.fields.solution_2d)
            
            print(thetis_obj.callbacks['export']['turbine'].integrated_power[0])
            solution = thetis_obj.fields.solution_2d
            # ic[field]=solution



            # timer end
            duration += perf_counter() # ej321 - TIMING
            # log time:
            logging.info(f'TIMING, forward_solve, {duration:.6f}')  # ej321 - TIMING
            print(f'forward_solve time taken: {duration:.6f} seconds')  # ej321 - TIMING
            
            # get model specific features
            self.get_model_features(self[index])
            # print(f'\n\n IN SOLVER FEATURES : {self.model_features.keys()}')

            # Solve the nonlinear shallow water equations.
            yield
            return {field: solution}

        return solver


    @staticmethod
    def steady_qoi_form(self, index, solution):
        thetis_obj=self.manage_thetis_object(self[index])



        rho = fd.Constant(self.params["density"])
        Ct = self.corrected_thrust_coefficient
        At = self.swept_area
        Cd = 0.5 * Ct * At * self.turbine_density(self[index])
        tags = self.turbine_ids
        u_, eta = solution.subfunctions #ej321
        # print(f'\n\n TEST JOE QOI: {rho} {At} {Ct} {fd.assemble(self.turbine_density(self[index])*fd.dx)}')
        print(f'\n\n TEST JOE QOI:{fd.assemble(sum([rho * Cd * pow(fd.dot(u_, u_), 1.5) * fd.dx(tag) for tag in tags]))}')
        
        # test out the power_output variable from Thetis
        tfarm = [farm.power_output(
        thetis_obj.callbacks['export']['turbine'].uv,
        thetis_obj.callbacks['export']['turbine'].depth)
        * (1030.0 * 1.0e-06)
        for farm in thetis_obj.callbacks['export']['turbine'].farms]
        print(f"FARM (power output method): {tfarm}")

        pfarm1 = [
            fd.assemble(farm.turbine.power(thetis_obj.callbacks['export']['turbine'].uv, 
                               thetis_obj.callbacks['export']['turbine'].depth)
             * farm.turbine_density
             * rho
             * fd.dx(tag)) for farm,tag in zip(thetis_obj.callbacks['export']['turbine'].farms, tags)]

        rfarm1 = [fd.assemble(
            farm.turbine.power(thetis_obj.callbacks['export']['turbine'].uv, 
                               thetis_obj.callbacks['export']['turbine'].depth)
             * farm.turbine_density
             * rho
             * farm.dx)
            for farm in thetis_obj.callbacks['export']['turbine'].farms]

        dfarm1 = fd.assemble(sum([
            farm.turbine.power(thetis_obj.callbacks['export']['turbine'].uv, 
                               thetis_obj.callbacks['export']['turbine'].depth)
             * farm.turbine_density
             * rho
              * fd.dx # test out if the farm.dx is the issue
            for farm in thetis_obj.callbacks['export']['turbine'].farms]))

        print(f'joe fd.dx(tag) area: {pfarm1 }, farm.dx area: {rfarm1}, fd.dx area: {dfarm1}')

       
        
        return sum([
            farm.turbine.power(thetis_obj.callbacks['export']['turbine'].uv, 
                               thetis_obj.callbacks['export']['turbine'].depth)
             * farm.turbine_density
            #  *fd.Constant(0.001) # convert to MW
            * rho # consistency with Joe's code
             * fd.dx
            for farm in thetis_obj.callbacks['export']['turbine'].farms]) 
    

            
        
    def calc_qoi(self,index,solutions):

        field= self.params["field"]
        solution = solutions[field]["forward"][-1][index]
        
        return self.steady_qoi_form(self, index, solution)


    @gol_adj.annotate_qoi
    def get_qoi(self, i):
    # def get_qoi(self, solutions, i):

        def qoi():
            thetis_obj=self.manage_thetis_object(self[i])          
            rho = fd.Constant(self.params["density"])
            return sum([
            farm.turbine.power(thetis_obj.callbacks['export']['turbine'].uv, 
                               thetis_obj.callbacks['export']['turbine'].depth)
             * farm.turbine_density
            #  *fd.Constant(0.001) # convert to MW
            * rho # consistency with Joe's code
             * fd.dx
            for farm in thetis_obj.callbacks['export']['turbine'].farms]) 

        return qoi

