from nn_adapt.solving import *


@PETSc.Log.EventDecorator('nn_adapt.get_hessians')
def get_hessians(f, **kwargs):
    """
    Compute Hessians for each component of
    a :class:`Function`.

    Any keyword arguments are passed to
    ``recover_hessian``.

    :arg f: the function
    :return: list of Hessians of each
        component
    """
    kwargs.setdefault('method', 'L2')
    return [
        hessian_metric(recover_hessian(fij, **kwargs))
        for i, fi in split_into_scalars(f).items()
        for fij in fi
    ]


@PETSc.Log.EventDecorator('nn_adapt.go_metric')
def go_metric(mesh, config, enrichment_method='h', target_complexity=4000.0,
              average=True, interpolant='L2', anisotropic=False, retall=False):
    """
    Compute an anisotropic goal-oriented
    metric field, based on a mesh and
    a configuration file.

    :arg mesh: input mesh
    :arg config: configuration file, which
        specifies the PDE and QoI
    :kwarg enrichment_method: how to enrich the
        finite element space?
    :kwarg target_complexity: target complexity
        of the goal-oriented metric
    :kwarg average: should the Hessian components
        be combined using averaging (or intersection)?
    :kwarg interpolant: which method to use to
        interpolate into the target space?
    :kwarg anisotropic: toggle isotropic vs.
        anisotropic metric
    :kwarg retall: if ``True``, the error indicator,
        forward solution, adjoint solution, enriched
        adjoint solution and :class:`GoalOrientedMeshSeq`
        are returned, in addition to the metric
    """
    dwr, fwd_sol, adj_sol, dwr_plus, adj_sol_plus, mesh_seq = indicate_errors(
        mesh, config, enrichment_method=enrichment_method, retall=True
    )
    if anisotropic:
        # TODO: It should be fwd_sol, not adj_sol
        hessian = combine_metrics(*get_hessians(adj_sol), average=average)
    else:
        hessian = None
    metric = anisotropic_metric(
        dwr, hessian=hessian,
        target_complexity=target_complexity,
        target_space=TensorFunctionSpace(mesh, 'DG', 0),
        interpolant=interpolant
    )
    if retall:
        return metric, dwr, fwd_sol, adj_sol, dwr_plus, adj_sol_plus, mesh_seq
    else:
        return metric
