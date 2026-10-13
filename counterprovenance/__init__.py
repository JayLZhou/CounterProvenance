"""CounterProvenance: exact provenance indexing for tree-ensemble counterfactual queries.

Typical use::

    from counterprovenance import bind, from_model, ProvenanceIndex

    ensemble = from_model(fitted_xgboost_or_lightgbm_or_sklearn_model)
    bound = bind(ensemble, unit_row, axes=[0, 3], noise=abducted_noise)
    index = ProvenanceIndex(bound)            # or autotune(bound, sample_points)
    answer = index.query([t0, t3])            # do(X0 = t0, X3 = t3)
    answer.value                              # exact counterfactual value
    answer.labels()                           # exact provenance: (tree, leaf) multiset
"""

from .cellindex import LAYOUTS, CellIndex, CellStats, Piece, iip_bytes, iip_records
from .ensemble import (
    BoundEnsemble,
    LeafBox,
    Slabs,
    Tree,
    TreeEnsemble,
    bind,
    from_lightgbm,
    from_model,
    from_sklearn_forest,
    from_sklearn_gbr,
    from_sklearn_hgb,
    from_xgboost,
)
from .index import Answer, IndexStats, ProvenanceIndex, TuneResult, autotune
from .overlay import FaceEdge, Overlay
from .persistence import INACTIVE, PersistentFactorTree, ancestor_union_size
from .plan import (
    Walk,
    best_walk,
    boustrophedon,
    edge_weights,
    euler_walk,
    greedy_dfs_forest,
    path_walk,
    shortcut_walk,
    spanning_forest,
)
from .semiring import CircuitArena

__all__ = [
    "Answer",
    "BoundEnsemble",
    "CellIndex",
    "CellStats",
    "CircuitArena",
    "FaceEdge",
    "INACTIVE",
    "IndexStats",
    "LAYOUTS",
    "LeafBox",
    "Overlay",
    "PersistentFactorTree",
    "Piece",
    "ProvenanceIndex",
    "Slabs",
    "Tree",
    "TreeEnsemble",
    "TuneResult",
    "Walk",
    "ancestor_union_size",
    "autotune",
    "bind",
    "best_walk",
    "boustrophedon",
    "edge_weights",
    "euler_walk",
    "from_lightgbm",
    "from_model",
    "from_sklearn_forest",
    "from_sklearn_gbr",
    "from_sklearn_hgb",
    "from_xgboost",
    "greedy_dfs_forest",
    "iip_bytes",
    "iip_records",
    "path_walk",
    "shortcut_walk",
    "spanning_forest",
]
