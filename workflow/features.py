import firedrake as fd
from collections.abc import Iterable
import animate as ani
from goalie import get_dwr_indicator as get_dwr_indicator
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
import ufl
import numpy as np
from firedrake import op2
import pickle

import torch # for ML implimentation #TODO: move this somewhere else?
import os # temp

__all__ = ["split_into_scalars", "get_hessians", "extract_components", 
"get_values_at_elements", "get_values_at_centroids", "get_tensor_at_centroids",
"extract_array", "get_mesh_info", "coarse_dwr_indicator", "extract_mesh_features",
"extract_coarse_dwr_features","proc_data_item",
"gnn_indicator_fit", "gnn_noadj_indicator_fit", "mlp_indicator_fit", "joe_indicator_fit"]

def split_into_scalars(f):
    """
    Given a :class:`Function`, split it into
    components from its constituent scalar
    spaces.

    If it is not mixed then no splitting is
    required.

    :arg f: the mixed :class:`Function`
    :return: a dictionary containing the
        nested structure of the mixed function
    """

    V = f.function_space()
    if V.value_size > 1:
        if not isinstance(V.node_count, Iterable):
            assert len(V.shape) == 1, "Tensor spaces not supported"
            el = V.ufl_element()
            fs = fd.FunctionSpace(V.mesh(), el.family(), el.degree())
            return {0: [fd.assemble(interpolate(f[i], fs)) for i in range(V.shape[0])]}
        subspaces = [V.sub(i) for i in range(len(V.node_count))]
        ret = {}
        for i, (Vi, fi) in enumerate(zip(subspaces, f.subfunctions)): #ej321 use subfuctions
            if len(Vi.shape) == 0:
                ret[i] = [fi]
            else:
                assert len(Vi.shape) == 1, "Tensor spaces not supported"
                el = Vi.ufl_element()
                fs = fd.FunctionSpace(V.mesh(), el.family(), el.degree())
                ret[i] = [fd.assemble(interpolate(fi[j], fs)) for j in range(Vi.shape[0])]
        return ret
    else:
        return {0: [f]}

def get_hessians(f, metric_parameters=None, **kwargs):
    """
    Compute Hessians for each component of
    a :class:`Function`.

    Any keyword arguments are passed to
    ``recover_hessian``.

    :arg f: the function
    :return: list of Hessians of each
        component
    """
    metrics = []
    for i, fi in split_into_scalars(f).items():
        for j,fij in enumerate(fi):

            M = ani.metric.RiemannianMetric(f.function_space().mesh())

            #add set parameters for min parameters to run
            if metric_parameters is not None:
                M.set_parameters(metric_parameters)
            else:
                M.set_parameters({
                    'dm_plex_metric_target_complexity':4000, # set quite complex
                    'dm_plex_metric_p': np.inf
                    })
            M.compute_hessian(fij, **kwargs)
#             M.enforce_spd(restrict_sizes=True) # from Joe's workflow - trying to match - makes a big difference
            M.normalise(restrict_sizes=False, restrict_anisotropy=False) # add explicit defaults so no enforce constraints done
            metrics.append(M)
            # QC:
            # File(f"/data0/firedrake_mar2023/src/nn_adapt/examples/steady_turbine/outputs/aligned/GO/hessian_loop.pvd").write(metric)
            # File(f"/data0/firedrake_mar2023/src/nn_adapt/examples/steady_turbine/outputs/aligned/GO/fij_loop.pvd").write(fij)
            # breakpoint()

    return metrics

def extract_components(matrix):
    r"""
    Extract components of a matrix that describe its
    size, orientation and shape.

    The latter two components are combined in such
    a way that we avoid errors relating to arguments
    zero and :math:`2\pi` being equal.
    """
    matrix_p0 = ani.metric.P0Metric(matrix) #call P0 class wrapper for RiemannianMetric
    density, quotients, evecs = matrix_p0.density_and_quotients(reorder=True)
    fs = density.function_space()
    ar = fd.assemble(interpolate(ufl.sqrt(quotients[1]), fs))
    armin = ar.vector().gather().min()
    #TODO: one 0.99999999 possible - needed edge case?
    if np.isclose(armin, 1.0, rtol=1e-08, atol=1e-08, equal_nan=False):
        armin = 1.0
    assert armin >= 1.0, f"An element has aspect ratio is less than one ({armin})"
    theta = fd.assemble(interpolate(ufl.atan(evecs[1, 1] / evecs[1, 0]), fs))
    h1 = fd.assemble(interpolate(ufl.cos(theta) ** 2 / ar + ufl.sin(theta) ** 2 * ar, fs))
    h2 = fd.assemble(interpolate((1 / ar - ar) * ufl.sin(theta) * ufl.cos(theta), fs))
    return density, h1, h2

def get_values_at_elements(f):
    """
    Extract the values for all degrees of freedom associated
    with each element.

    :arg f: some :class:`Function`
    :return: a vector :class:`Function` holding all DoFs of `f`
    """
    fs = f.function_space()
    mesh = fs.mesh()
    dim = mesh.topological_dimension()
    if dim == 2:
        assert fs.ufl_element().cell == ufl.triangle, "Simplex meshes only"
    elif dim == 3:
        assert fs.ufl_element().cell == ufl.tetrahedron, "Simplex meshes only"
    else:
        raise ValueError(f"Dimension {dim} not supported")
    el = fs.ufl_element()
    if el.sub_elements == []:
        p = el.degree()
        size = fs.value_size * (p + 1) * (p + 2) // 2
    else:
        size = 0
        for sel in el.sub_elements:
            p = sel.degree()
            size += sel.value_size * (p + 1) * (p + 2) // 2
    P0_vec = fd.VectorFunctionSpace(mesh, "DG", 0, dim=size)
    values = fd.Function(P0_vec)
    keys = {"vertexwise": (f, op2.READ), "elementwise": (values, op2.INC)}
    # original code - C style which is depreciated in firedrake
    # kernel = "for (int i=0; i < vertexwise.dofs; i++) elementwise[i] += vertexwise[i];" 
    # firedrake.par_loop(kernel, ufl.dx, keys)

    # loopy format refactoring like with clement interpolation in pyroteus
    domain_op2 = '{[i]: 0 <= i < vertexwise.dofs}' 
    instructions = '''
    for i
        elementwise[i] = elementwise[i] + vertexwise[i]
    end
    '''
    fd.par_loop(
        (domain_op2,instructions), ufl.dx, 
        keys
    # ,is_loopy_kernel=True # depreciated, throws warning
    )
    return values

def get_values_at_centroids(f):
    """
    Extract the values for the function at each element centroid,
    along with all derivatives up to the :math:`p^{th}`, where
    :math:`p` is the polynomial degree.

    :arg f: some :class:`Function`
    :return: a vector :class:`Function` holding all DoFs of `f`
    """
    fs = f.function_space()
    mesh = fs.mesh()
    dim = mesh.topological_dimension()
    if dim == 2:
        assert fs.ufl_element().cell == ufl.triangle, "Simplex meshes only"
    elif dim == 3:
        assert fs.ufl_element().cell == ufl.tetrahedron, "Simplex meshes only"
    else:
        raise ValueError(f"Dimension {dim} not supported")
    el = fs.ufl_element()
    if el.sub_elements == []:
        p = el.degree()
        degrees = [p]
        size = fs.value_size * (p + 1) * (p + 2) // 2
        funcs = [f]
    else:
        size = 0
        degrees = [sel.degree() for sel in el.sub_elements]
        for sel, p in zip(el.sub_elements, degrees):
            size += sel.value_size * (p + 1) * (p + 2) // 2
        funcs = f
    values = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=size))
    P0 = fd.FunctionSpace(mesh, "DG", 0)
    P0_vec = fd.VectorFunctionSpace(mesh, "DG", 0)
    P0_ten = fd.TensorFunctionSpace(mesh, "DG", 0)
    i = 0
    for func, p in zip(funcs, degrees):
        values.dat.data[:, i] = fd.project(func, P0).dat.data_ro
        i += 1
        if p == 0:
            continue
        g = fd.project(ufl.grad(func), P0_vec)
        values.dat.data[:, i] = g.dat.data_ro[:, 0]
        values.dat.data[:, i + 1] = g.dat.data_ro[:, 1]
        i += 2
        if p == 1:
            continue
        H = fd.project(ufl.grad(ufl.grad(func)), P0_ten)
        values.dat.data[:, i] = H.dat.data_ro[:, 0, 0]
        values.dat.data[:, i + 1] = 0.5 * (
            H.dat.data_ro[:, 0, 1] + H.dat.data_ro[:, 1, 0]
        )
        values.dat.data[:, i + 2] = H.dat.data_ro[:, 1, 1]
        i += 3
        if p > 2:
            raise NotImplementedError(
                "Polynomial degrees greater than 2 not yet considered"
            )
    return values

@PETSc.Log.EventDecorator("Extract tensor at centroids")
def get_tensor_at_centroids(f):
    """
    :arg f: some :class:`Function`
    :return: a vector :class:`Function` holding all DoFs of `f`
    """
    fs = f.function_space()
    mesh = fs.mesh()
    dim = mesh.topological_dimension()
    if dim == 2:
        assert fs.ufl_element().cell == ufl.triangle, "Simplex meshes only"
    elif dim == 3:
        assert fs.ufl_element().cell == ufl.tetrahedron, "Simplex meshes only"
    else:
        raise ValueError(f"Dimension {dim} not supported")
    el = fs.ufl_element()
    if el.sub_elements == []:
        p = el.degree()
        degrees = [p]
        size = fs.value_size * (p + 1) * (p + 2) // 2
        funcs = [f]
    else:
        size = 0
        degrees = [sel.degree() for sel in el.sub_elements]
        for sel, p in zip(el.sub_elements, degrees):
            size += sel.value_size * (p + 1) * (p + 2) // 2
        funcs = f
    values = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 0, dim=size))
    P0 = fd.FunctionSpace(mesh, "DG", 0)
    P0_vec = fd.VectorFunctionSpace(mesh, "DG", 0)
    P0_ten = fd.TensorFunctionSpace(mesh, "DG", 0)
    i = 0

    func=funcs # temp
    # if (funcs.len>1):
    #     for func in funcs:
    #         values.dat.data[:, i] = fd.project(func, P0).dat.data_ro
    #         i += 1
    #         if p == 0:
    #             continue
    #         g = fd.project(ufl.grad(func), P0_vec)
    #         values.dat.data[:, i] = g.dat.data_ro[:, 0]
    #         values.dat.data[:, i + 1] = g.dat.data_ro[:, 1]
    #         i += 2
    #         if p == 1:
    #             continue
    #         H = fd.project(ufl.grad(ufl.grad(func)), P0_ten)
    func_len = func.dat.data_ro[:, 0, 0].shape[0] # added to fit shape, HACKY
    values.dat.data[:func_len, i] = func.dat.data_ro[:, 0, 0]
    values.dat.data[:func_len, i + 1] = 0.5 * (
        func.dat.data_ro[:, 0, 1] + func.dat.data_ro[:, 1, 0]
    )
    values.dat.data[:func_len, i + 2] = func.dat.data_ro[:, 1, 1]
    i += 3
    # if degrees > 2:
    #     raise NotImplementedError(
    #         "Polynomial degrees greater than 2 not yet considered"
    #     )
    return values.dat.data

def extract_array(f, mesh=None, centroid=False, project=False):
    r"""
    Extract a cell-wise data array from a :class:`Constant` or
    :class:`Function`.

    For constants and scalar fields, this will be an :math:`n\times 1`
    array, where :math:`n` is the number of mesh elements. For a mixed
    field with :math:`m` components, it will be :math:`n\times m`.

    :arg f: the :class:`Constant` or :class:`Function`
    :kwarg mesh: the underlying :class:`MeshGeometry`
    :kwarg project: if ``True``, project the field into
        :math:`\mathbb P0` space
    """
    mesh = mesh or f.ufl_domain()
    if isinstance(f, fd.Constant):
        ones = np.ones(mesh.num_cells())
        assert len(f.values()) == 1
        return f.values()[0] * ones
    elif not isinstance(f, fd.Function):
        raise ValueError(f"Unexpected input type {type(f)}")
    if project:
        if len(f.function_space().shape) > 0:
            raise NotImplementedError("Can currently only project scalar fields")  # TODO
        element = f.ufl_element()
        if (element.family(), element.degree()) != ("Discontinuous Lagrange", 0):
            P0 = fd.FunctionSpace(mesh, "DG", 0)
            f = fd.project(f, P0)
    s = sum([fi for i, fi in split_into_scalars(f).items()], start=[])
    get = get_values_at_centroids if centroid else get_values_at_elements
    if len(s) == 1:
        # print(f'extract array len ==1')
        return get(s[0]).dat.data
    else:
        # print(f'extract array len <>1')
        return np.hstack([get(si).dat.data for si in s])
    
def get_mesh_info(fwd_sol):
    
    mesh = fwd_sol.function_space().mesh()

    nodes = []
    cellinfo={}

    q=mesh.topology_dm.getCoordinates().array
    coords = [(a,b) for (a,b) in zip(q[::2],q[1::2])]
    for i,a in enumerate(mesh.cell_closure):
        _element={}

        _element["element_id"] = a[-1]

        cell_con=[]
        _element["element_edges"] =mesh.topology_dm.getCone(a[-1])
        for e in _element["element_edges"]:
            g = list(mesh.topology_dm.getSupport(e))
            if len(g)>1:
                cell_con.append((g[0],g[1]))
        _element["element_links"] = list(set(cell_con))

        connect=[]
        vertices =[v for v in a[:-1] if v not in _element["element_edges"]]
        for v in vertices:
            if mesh.topology_dm.getConeSize(v)==0:
                for c in mesh.topology_dm.getSupport(v):
                    d = mesh.topology_dm.getCone(c)
                    connect.append((d[0],d[1]))
                connect = list(set(connect))
  
        _element["element_vertices"] = vertices
        _element["vertices_links"] = connect
        
        lcoords = []
        for b in _element["element_vertices"]:
            lcoords.append(coords[b-mesh.num_cells()])
        centr=(np.mean([x for x,y in lcoords]),np.mean([y for x,y in lcoords]))
        _element["element_centroid_xy"]= centr    
        _element["element_vertices_xy"]= lcoords

        cellinfo[i]=_element

    return cellinfo

def coarse_dwr_indicator(mesh_seq, fwd_sol,adj_sol, index=0):
    r"""
    Evaluate the DWR error indicator as a :math:`\mathbb P0` field.
    Assumes steady state - first mesh is hard coded

    :arg mesh: the current mesh
    :arg q: the forward solution, transferred into enriched space
    :arg q_star: the adjoint solution in enriched space
    """
    mesh = fwd_sol.function_space().mesh()

    # Extract indicator in enriched space
    field=mesh_seq.params["field"]

    F = mesh_seq.forms[field]

    V=fwd_sol.function_space()

    # QC: 
    print(f'V: {fwd_sol.function_space()}')
    # print(f'adjoint : {adj_sol}')
    dwr_star = get_dwr_indicator(F, adj_sol, test_space=V)

    # Project down to base space
    P0 = fd.FunctionSpace(mesh, "DG", 0)
    dwr_coarse = fd.project(dwr_star, P0)
    dwr_coarse.interpolate(abs(dwr_coarse))
    return dwr_coarse

def extract_mesh_features(fwd_sol):
    
    # https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html
    mesh = fwd_sol.function_space().mesh()

    # Features describing the mesh element
    with PETSc.Log.Event("Analyse element"):
        P0_ten = fd.TensorFunctionSpace(mesh, "DG", 0) # for steady state???
        # P0_ten = fd.TensorFunctionSpace(mesh, "CG", 1)
        # Element size, orientation and shape
        J = ufl.Jacobian(mesh)
        JTJ = fd.assemble(interpolate(fd.dot(fd.transpose(J), J), P0_ten))
        
        d, h1, h2 = (extract_array(p) for p in extract_components(JTJ))

        # Is the element on the boundary?
        # TODO: trying this as alter to passing dwr_coarse
        P0 = fd.FunctionSpace(mesh, "DG", 0)
        # p0test = fd.TestFunction(dwr_coarse.function_space())
        p0test = fd.TestFunction(P0)
        bnd = fd.assemble(p0test * ufl.ds).dat.data

        # returns arrays of mesh_d, mesh_h1, mesh_h2 and mesh bnd
        return d, h1, h2, bnd

def extract_coarse_dwr_features(mesh_seq, fwd_sol, adj_sol, index=0):
    dwr_coarse = coarse_dwr_indicator(mesh_seq, fwd_sol, adj_sol, index)
    return extract_array(dwr_coarse)


def proc_data_item(data_item, include_global_attrs=True, use_pos=True, no_adjoint=False,
                   attr_names=["estimator_coarse","physics_viscosity", "physics_drag", "physics_bathymetry", 
                               "mesh_d", "mesh_h1", "mesh_h2", "mesh_bnd", "forward_dofs", "adjoint_dofs"]): 
    """
    Processes an output from Firedrake nn-adapt to a dictionary to be fed to a PyG dataset. 
    Function from Siyi gnn_adapt repository with minor modifications

    Args:
        data_item (dict): nn-adapt output of a single mesh.
        include_global_attrs (bool, optional): _description_. Defaults to True.
        attr_names (list, optional): _description_. Defaults to ["estimator_coarse", "physics_drag", "physics_viscosity", "physics_bathymetry", "mesh_d", "mesh_h1", "mesh_h2", "mesh_bnd", "forward_dofs", "adjoint_dofs"].

    Returns:
        dict: a more PyG-friendly dictionary
    """
    # instance of proc dictionary
    proc_dict = {}
    # the number of cells should be equal to the cell_info list
    num_cells = len(data_item["cell_info"])
    
    # placeholders for features
    cell_idx_dict = {}
    list_edge_idx_orig = []
    list_edge_idx_ordered = []
    list_cell_pos = []
    
    # populate cell positions and edge ids lists
    for key, value in data_item["cell_info"].items():
        # maps firedrake_idx to index in feature matrix
        cell_idx_dict[value["element_id"]] = key
        # centroid position of each element
        list_cell_pos.append(np.array(value["element_centroid_xy"]))
        # list of all edges between elements in the mesh
        for i in range(len(value["element_links"])):
            list_edge_idx_orig.append(np.array(value["element_links"][i]))        
    
    # for each edge between elements 
    #  - renumbers the element id to sequential series
    #  - adds to a new ordered list for easier import to pytorch
    for edge in list_edge_idx_orig:
        ordered_edge = [cell_idx_dict[edge[0]], cell_idx_dict[edge[1]]]
        list_edge_idx_ordered.append(ordered_edge)
    
    if no_adjoint and "adjoint_dofs" in attr_names:
        attr_names.remove("adjoint_dofs")

    # account for format of general solution saved in pickle
    if len(data_item["forward_dofs"]):
        data_item['forward_dofs'] = data_item['forward_dofs']['0.0']
    else:
        print(f'not implimented for multiple solutions')

    print(attr_names)
    # account for format in saved pickle file
    # TODO - make this general or figure out where tuple is introduced
    if "physics_viscosity" in attr_names:
        data_item['physics_viscosity'] = data_item['physics_viscosity'][0]
    if "physics_drag" in attr_names:
        data_item['physics_drag'] = data_item['physics_drag'][0]
    if "physics_bathymetry" in attr_names:
        data_item['physics_bathymetry'] = data_item['physics_bathymetry'][0]

    # convert the cell attributes into a column stack
    proc_dict["cell_attrs"] = np.column_stack([data_item[key] for key in attr_names])
    
    # if including global attributes, create additional one to many columns
    if include_global_attrs:
        proc_dict["cell_attrs"] = np.column_stack([
            proc_dict["cell_attrs"], 
            np.repeat(data_item["global_inflow_speed"], repeats=num_cells)
            ])
    
    proc_dict["cell_pos"] = np.array(list_cell_pos)
    proc_dict["edge_index_ordered"] = np.array(list_edge_idx_ordered)

    # if using the position
    #  - create a torch tensor with the cell position and associated attributes
    #  - [x, y, f1, f2, f3, ....]
    # else, just aggregate the associated attributes
    #  - [f1, f2, f3, ....]
    if use_pos:
        node_attrs = torch.as_tensor(np.column_stack(
            [proc_dict["cell_pos"], 
            proc_dict["cell_attrs"]
            ]), dtype=torch.float32)
    else:
        node_attrs = torch.as_tensor(
            proc_dict["cell_attrs"],
            dtype=torch.float32)
    
    # create a torch tensor with the edge renumbered element edge indexes
    edge_index = torch.as_tensor(proc_dict["edge_index_ordered"].T, dtype=torch.long)
    return node_attrs, edge_index
    

def gnn_indicator_fit(features, mesh):
    # TODO: setup now for steady, not unsteady problems

    # QC features input
    # print(f'gnn indicator fit - input features: {features.keys()}')

    # reformat for gnn - node and edge pytorch tensor objects
    node_attrs, edge_index = proc_data_item(features)

    # QC features reformated for gnn
    # print(f'node size {node_attrs.shape}, edge size {edge_index.shape}')

    # QC current folder:
    print(f'current director for graph models root path: {os.getcwd()}')

    # set root location for the saved model
    # TODO: make this not hardcoded or move 
    # localpath = '/home/phd01/nn_adapt'
    localpath = '/data0'
        
    # load model from pytorch
    gnn_model = torch.jit.load(f"{localpath}/trained_models/graphsage_5e-5_dwr_cap2.5.pt")
    normalise_func = torch.jit.load(f"{localpath}/trained_models/normalise_attrs_2196.pt")
    denormalise_func = torch.jit.load(f"{localpath}/trained_models/denormalise_targets_2196.pt")
    
    # QC function imports working
    # print(f'in the gnn indictor fit {gnn_model}')

    # Run GNN model
    x_normalised = normalise_func(node_attrs)
    pred_normalised = gnn_model(x_normalised, edge_index).clip(min=0.0, max=1.0)
    pred = denormalise_func(pred_normalised).clip(min=0.0)

    P0 = fd.FunctionSpace(mesh, "DG", 0)
    dwr = fd.Function(P0)
    dwr.dat.data[:] = pred.detach().numpy()
    
    return dwr


def gnn_noadj_indicator_fit(features, mesh):
    # TODO: setup now for steady, not unsteady problems

    # QC features input
    # print(f'gnn indicator fit - input features: {features.keys()}')

    # reformat for gnn - node and edge pytorch tensor objects
    node_attrs, edge_index = proc_data_item(features, use_pos=True, no_adjoint=True)

    # QC features reformated for gnn
    # print(f'node size {node_attrs.shape}, edge size {edge_index.shape}')

    # set root location for the saved model
    # TODO: make this not hardcoded or move 
    # localpath = '/home/phd01/nn_adapt'
    localpath = '/data0'
        
    # load model from pytorch
    gnn_noadj_model = torch.jit.load(f"{localpath}/trained_models/graphsage_noadj.pt")
    normalise_func = torch.jit.load(f"{localpath}/trained_models/normalise_attrs_2196_noadj.pt")
    denormalise_func = torch.jit.load(f"{localpath}/trained_models/denormalise_targets_2196_noadj.pt")
    
    # QC function imports working
    # print(f'in the gnn indictor fit {gnn_model}')

    # Run GNN model
    x_normalised = normalise_func(node_attrs)
    pred_normalised = gnn_noadj_model(x_normalised, edge_index).clip(min=0.0, max=1.0)
    pred = denormalise_func(pred_normalised).clip(min=0.0)

    P0 = fd.FunctionSpace(mesh, "DG", 0)
    dwr = fd.Function(P0)
    dwr.dat.data[:] = pred.detach().numpy()
    
    return dwr


def mlp_indicator_fit(features, mesh):
    # TODO: setup now for steady, not unsteady problems

    # QC features input
    print(f'gnn indicator fit - input features: {features.keys()}')

    # reformat for gnn - node and edge pytorch tensor objects
    node_attrs, edge_index = proc_data_item(features)

    # QC features reformated for gnn
    print(f'node size {node_attrs.shape}, edge size {edge_index.shape}')

    # set root location for the saved model
    # TODO: make this not hardcoded or move 
    # localpath = '/home/phd01/nn_adapt'
    localpath = '/data0'
        
    # load model from pytorch
    mlp_model = torch.jit.load(f"{localpath}/trained_models/mlp.pt")
    normalise_func = torch.jit.load(f"{localpath}/trained_models/normalise_attrs_2196.pt") # TODO: Check with Siyi 
    denormalise_func = torch.jit.load(f"{localpath}/trained_models/denormalise_targets_2196.pt") # TODO: Check with Siyi 
    
    # QC function imports working
    print(f'in the gnn indictor fit {mlp_model}')

    # Run GNN model
    x_normalised = normalise_func(node_attrs)
    pred_normalised = mlp_model(x_normalised).clip(min=0.0, max=1.0)
    pred = denormalise_func(pred_normalised).clip(min=0.0)

    P0 = fd.FunctionSpace(mesh, "DG", 0)
    dwr = fd.Function(P0)
    dwr.dat.data[:] = pred.detach().numpy()
    
    return dwr

def joe_indicator_fit(features, mesh):
    # TODO: setup now for steady, not unsteady problems

    # QC features input
    # print(f'gnn indicator fit - input features: {features.keys()}')

    # reformat for gnn - node and edge pytorch tensor objects
    node_attrs, edge_index = proc_data_item(features)

    # QC features reformated for gnn
    # print(f'node size {node_attrs.shape}, edge size {edge_index.shape}')

    # set root location for the saved model
    # TODO: make this not hardcoded or move 
    # localpath = '/home/phd01/nn_adapt'
    localpath = '/data0'
        
    # load model from pytorch
    mlp_model = torch.jit.load(f"{localpath}/trained_models/mlp.pt")
    normalise_func = torch.jit.load(f"{localpath}/trained_models/normalise_attrs_2196.pt") # TODO: Check with Siyi 
    denormalise_func = torch.jit.load(f"{localpath}/trained_models/denormalise_targets_2196.pt") # TODO: Check with Siyi 
    
    # QC function imports working
    # print(f'in the gnn indictor fit {gnn_model}')

    # Run GNN model
    x_normalised = normalise_func(node_attrs)
    pred_normalised = mlp_model(x_normalised).clip(min=0.0, max=1.0)
    pred = denormalise_func(pred_normalised).clip(min=0.0)

    P0 = fd.FunctionSpace(mesh, "DG", 0)
    dwr = fd.Function(P0)
    dwr.dat.data[:] = pred.detach().numpy()
    
    return dwr