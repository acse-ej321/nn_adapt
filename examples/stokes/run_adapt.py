from nn_adapt import *
import argparse
import importlib


set_log_level(ERROR)

# Parse for test case and number of refinements
parser = argparse.ArgumentParser()
parser.add_argument('test_case', help='The configuration file number')
parser.add_argument('-miniter', help='Minimum number of iterations (default 3)')
parser.add_argument('-maxiter', help='Maximum number of iterations (default 35)')
parser.add_argument('-qoi_rtol', help='Relative tolerance for QoI (default 0.005)')
parser.add_argument('-element_rtol', help='Relative tolerance for element count (default 0.005)')
parser.add_argument('-estimator_rtol', help='Relative tolerance for error estimator (default 0.005)')
parser.add_argument('-target_complexity', help='Target metric complexity (default 4000.0)')
parsed_args, unknown_args = parser.parse_known_args()
test_case = int(parsed_args.test_case)
assert test_case in [0, 1, 2, 3, 4]
miniter = int(parsed_args.miniter or 3)
assert miniter >= 0
maxiter = int(parsed_args.maxiter or 35)
assert maxiter >= miniter
qoi_rtol = float(parsed_args.qoi_rtol or 0.005)
assert qoi_rtol > 0.0
element_rtol = float(parsed_args.element_rtol or 0.005)
assert element_rtol > 0.0
estimator_rtol = float(parsed_args.estimator_rtol or 0.005)
assert estimator_rtol > 0.0
target_complexity = float(parsed_args.target_complexity or 4000.0)
assert target_complexity > 0.0
# p = 1

# Setup
config = importlib.import_module(f'config{test_case}')
field = config.fields[0]
plex = PETSc.DMPlex().create()
plex.createFromFile(os.path.join(os.path.abspath(os.path.dirname(__file__)), f'meshes/{test_case}.h5'))
mesh = Mesh(plex)

# Run adaptation loop
kwargs = {
    'enrichment_method': 'h',
    'target_complexity': target_complexity,
    'average': True,
    'retall': True,
}
qoi_old = None
elements_old = mesh.num_cells()
estimator_old = None
converged_reason = None
fwd_file = File(f'outputs/go/forward{test_case}.pvd')
adj_file = File(f'outputs/go/adjoint{test_case}.pvd')
adj_plus_file = File(f'outputs/go/enriched_adjoint{test_case}.pvd')
ee_file = File(f'outputs/go/estimator{test_case}.pvd')
ee_plus_file = File(f'outputs/go/enriched_estimator{test_case}.pvd')
print('Mesh 0')
print(f'  Element count        = {elements_old}')
for fp_iteration in range(maxiter+1):

    # Compute goal-oriented metric
    p0metric, dwr, fwd_sol, adj_sol, dwr_plus, adj_sol_plus, mesh_seq = go_metric(mesh, config, **kwargs)
    fwd_file.write(*fwd_sol.split())
    adj_file.write(*adj_sol.split())
    adj_plus_file.write(*adj_sol_plus.split())
    ee_file.write(dwr)
    ee_plus_file.write(dwr_plus)

    # Check for QoI convergence
    qoi = mesh_seq.J
    print(f'  Quantity of Interest = {qoi}')
    if qoi_old is not None and fp_iteration >= miniter:
        if abs(qoi - qoi_old) < qoi_rtol*abs(qoi_old):
            converged_reason = 'QoI convergence'
            break
    qoi_old = qoi

    # Check for error estimator convergence
    estimator = dwr.vector().gather().sum()
    print(f'  Error estimator      = {estimator}')
    if estimator_old is not None and fp_iteration >= miniter:
        if abs(estimator - estimator_old) < estimator_rtol*abs(estimator_old):
            converged_reason = 'error estimator convergence'
            break
    estimator_old = estimator

    # Process metric
    P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
    p1metric = hessian_metric(clement_interpolant(p0metric))
    # space_normalise(p1metric, target_complexity, p)
    enforce_element_constraints(p1metric, 1.0e-05, 10.0, 1.0e+05)

    # Adapt the mesh and check for element count convergence
    mesh = adapt(mesh, p1metric)
    elements = mesh.num_cells()
    print(f'Mesh {fp_iteration+1}')
    print(f'  Element count        = {elements}')
    if fp_iteration >= miniter:
        if abs(elements - elements_old) < element_rtol*abs(elements_old):
            converged_reason = 'element count convergence'
            break
    elements_old = elements

    # Check for reaching maximum number of iterations
    if fp_iteration == maxiter:
        converged_reason = 'reaching maximum iteration count'
print(f'Terminated after {fp_iteration+1} iterations due to {converged_reason}')