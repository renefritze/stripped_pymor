import re

MANUAL_SKIPS = ('pymor.analyticalproblems.domaindescriptions.DomainDescription.dim',
                'pymor.analyticalproblems.domaindescriptions.DomainDescription.boundary_types',
                'pymor.analyticalproblems.expressions.Expression.shape',
                'pymor.core.base.BasicObject.name',
                'pymor.core.base.BasicObject.logging_disabled',
                'pymor.core.base.BasicObject.logger',
                'pymor.core.base.BasicObject.uid',
                'pymor.core.cache.CacheRegion.persistent',
                'pymor.core.cache.CacheableObject.cache_region',
                'pymor.core.cache.CacheableObject.cache_id',
                'pymor.core.defaults._default_container',
                'pymor.discretizers.builtin.grids.interfaces.ReferenceElement.dim',
                'pymor.discretizers.builtin.grids.interfaces.ReferenceElement.volume',
                'pymor.discretizers.builtin.grids.interfaces.BoundaryInfo.boundary_types',
                'pymor.discretizers.builtin.grids.referenceelements.Point.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Point.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Line.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Line.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Square.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Square.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Triangle.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Triangle.volume',
                'pymor.discretizers.builtin.grids.subgrid.SubGrid.parent_grid',
                'pymor.models.interface.Model.dim_output',
                'pymor.models.interface.Model.linear',
                'pymor.models.interface.Model.products',
                'pymor.operators.block.BlockOperatorBase.H',
                'pymor.operators.block.adjoint_type',
                'pymor.operators.interface.Operator.H',
                'pymor.parallel.interface.RemoteObject.removed',
                'pymor.operators.list.LinearComplexifiedListVectorArrayOperatorBase.linear',
                'pymor.parameters.base.Mu.parameters',
                'pymor.parameters.base.ParametricObject.parameters',
                'pymor.parameters.base.ParametricObject.parameters_own',
                'pymor.parameters.base.ParametricObject.parameters_internal',
                'pymor.parameters.base.ParametricObject.parameters_inherited',
                'pymor.parameters.base.ParametricObject.parametric',
                'pymor.models.interface.Model.solution_space',
                'pymor.operators.interface.Operator.solver_options',
                'pymor.operators.numpy.NumpyMatrixBasedOperator.sparse',
                'pymor.discretizers.builtin.fv.NonlinearReactionOperator.linear',
                'RENDER_VERTEX_SHADER',
                'RENDER_FRAGMENT_SHADER',
                'pymor.vectorarrays.interface.VectorArray.is_view',
                'pymor.vectorarrays.interface.VectorArray.dim',
                'pymor.vectorarrays.interface.VectorSpace.id',
                'pymor.vectorarrays.interface.VectorSpace.dim',
                'pymor.vectorarrays.interface.VectorSpace.is_scalar',
                )
SKIPS_RE = re.compile(r'(?:{})'.format('|'.join(map(re.escape, sorted(MANUAL_SKIPS, key=len, reverse=True)))))


def skip(app, what, name, obj, skip, options):
    try:
        if ":noindex:" in obj.docstring:
            print(f'HERE DO_SKIP {name}')
            return True
    except AttributeError:
        pass
    return SKIPS_RE.search(name)


def setup(app):
    app.connect('autoapi-skip-member', skip)
    return {'parallel_read_safe': True}
