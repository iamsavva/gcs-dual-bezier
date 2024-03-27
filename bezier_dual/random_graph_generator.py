import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    VPolytope,
)


from bezier_dual import (
    PolynomialDualGCS,
    QUADRATIC_COST,
    DualVertex,
    DualEdge,
    QUADRATIC_COST_AUGMENTED,
)
from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions


def random_uniform_graph_generator(
    options: ProgramOptions,
    use_bidirecitonal_edges=True,
    target_set_width=1.5,
    Q_terminal: npt.NDArray = np.eye(2) * 10,
    cost_function=QUADRATIC_COST,
    num_layers: int = 5,
    x_min: float = 0,
    x_max: float = 10,
    min_blank: float = 0.5,
    max_blank: float = 1.5,
    min_region: float = 0.5,
    max_region: float = 1.5,
    min_goal_blank: float = 1,
    max_goal_blank: float = 2,
    goal_num: int = 5,
    goal_uniform: bool = False,
    random_seed=1,
):
    gcs = PolynomialDualGCS(options)

    np.random.seed(random_seed)

    def box(a, b, layer, d=3):
        return Hyperrectangle([a, 2 * layer], [b, 2 * layer + d])

    ###############################################################
    # make vertices

    layers = []
    # add first layer
    start_vertex = gcs.AddVertex(
        "0-0", box(x_min, x_max, num_layers - 1), vertex_is_start=True
    )
    layers.append([start_vertex])

    # for every layer
    for n in range(1, num_layers - 1):
        layer = []
        x_now = 0.0
        k = n % 2
        index = 0
        while x_now < x_max:
            # make a skip
            if k % 2 == 0:
                x_now += np.random.uniform(min_blank, max_blank, 1)[0]
            else:
                width = np.random.uniform(min_region, max_region, 1)[0]
                v_name = str(n) + "-" + str(index)
                v = gcs.AddVertex(
                    v_name, box(x_now, min(x_now + width, x_max), num_layers - 1 - n)
                )
                layer.append(v)
                index += 1
                x_now += width
            k += 1
        layers.append(layer)

    # add target potential
    layer = []
    index = 0
    if goal_uniform:
        points = (
            (np.array(list(range(int(goal_num)))) + 0.5) * (x_max - x_min) / goal_num
        )
        for p in points:
            v_name = str(num_layers - 1) + "-" + str(index)
            v = gcs.AddVertex(
                v_name, box(p - target_set_width, p + target_set_width, 0)
            )
            layer.append(v)
            vt = gcs.AddTargetVertexWithQuadraticTerminalCost(
                v_name + "T",
                box(p - 0.1, p + 0.1, 0, 0.1),
                Q_terminal,
                np.array([p, 0]),
            )
            gcs.AddEdge(v, vt, cost_function)
            index += 1
    else:
        x_now = np.random.uniform(min_goal_blank, max_goal_blank, 1)[0]
        while x_now < x_max:
            v_name = str(num_layers - 1) + "-" + str(index)
            v = gcs.AddVertex(
                v_name, box(x_now - target_set_width, x_now + target_set_width, 0)
            )
            layer.append(v)
            vt = gcs.AddTargetVertexWithQuadraticTerminalCost(
                v_name + "T",
                box(x_now - 0.1, x_now + 0.1, 0, 0.1),
                Q_terminal,
                np.array([x_now, 0]),
            )
            gcs.AddEdge(v, vt, cost_function)
            x_now += np.random.uniform(min_goal_blank, max_goal_blank, 1)[0]
            index += 1
    layers.append(layer)

    ###############################################################
    # make edges
    for i, layer in enumerate(layers[:-1]):
        next_layer = layers[i + 1]
        for left_v in layer:
            for right_v in next_layer:
                if left_v.get_hpoly().IntersectsWith(right_v.get_hpoly()):
                    if use_bidirecitonal_edges:
                        gcs.AddBidirectionalEdge(left_v, right_v, cost_function)
                    else:
                        gcs.AddEdge(left_v, right_v, cost_function)

    # push up on start
    gcs.MaxCostOverABox(
        start_vertex, start_vertex.convex_set.lb(), start_vertex.convex_set.ub()
    )
    return gcs, layers, start_vertex
