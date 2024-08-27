import taichi as ti
from taichi.math import vec3, normalize
from primitives.aabb import AABB, union, union_p, intersect_bounds
from primitives.intersects import Intersection
from primitives.ray import Ray, spawn_ray
from utils.constants import INF, MAX_DEPTH



@ti.dataclass
class BVHNode:
    bounds: AABB
    first_prim_offset: ti.i32
    n_primitives: ti.i32
    child_0: ti.i32
    child_1: ti.i32
    split_axis: ti.i32

    @ti.func
    def init_leaf(self, first, n, box: ti.template()):
        self.first_prim_offset = first
        self.n_primitives = n
        self.bounds = box

    @ti.func
    def init_interior(self, axis, c0, c1, box: ti.template()):
        self.child_0 = c0
        self.child_1 = c1
        self.split_axis = axis
        self.bounds = box
        self.n_primitives = 0


@ti.dataclass
class LinearBVHNode:
    bounds: AABB
    primitives_offset: ti.i32
    second_child_offset: ti.i32
    n_primitives: ti.i32
    axis: ti.i32


@ti.dataclass
class BucketInfo:
    count: ti.i32
    bounds: AABB


@ti.dataclass
class Queue:
    item: ti.i32


@ti.dataclass
class BuildParams:
    n_triangles: ti.i32
    n_ordered_prims: ti.i32
    total_nodes: ti.i32
    split_method: ti.i32


@ti.func
def push(stack, stack_ptr, start, end, parent_idx, is_second_child):
    stack[stack_ptr[None], 0] = start
    stack[stack_ptr[None], 1] = end
    stack[stack_ptr[None], 2] = parent_idx
    stack[stack_ptr[None], 3] = is_second_child
    stack_ptr[None] += 1


@ti.func
def pop(stack, stack_ptr):
    stack_ptr[None] -= 1
    start = stack[stack_ptr[None], 0]
    end = stack[stack_ptr[None], 1]
    parent_idx = stack[stack_ptr[None], 2]
    is_second_child = stack[stack_ptr[None], 3]
    return start, end, parent_idx, is_second_child


@ti.kernel
def build_bvh(primitives: ti.template(),
              bvh_primitives: ti.template(),
              _start: ti.template(),
              _end: ti.template(),
              ordered_prims: ti.template(),
              nodes: ti.template(),
              total_nodes: ti.template(),
              split_method: ti.template(),
              stack: ti.template(),
              stack_ptr: ti.template(),
              ordered_prims_idx: ti.template(),
              costs: ti.template(),
              buckets: ti.template()) -> ti.i32:
    n_boxes = bvh_primitives.shape[0]
    max_prims_in_node = ti.max(4, ti.cast(0.1 * n_boxes, ti.i32))

    push(stack, stack_ptr, _start[None], _end[None], -1, 0)

    while stack_ptr[None] > 0:
        start, end, parent_idx, is_second_child = pop(stack, stack_ptr)

        current_node_idx = total_nodes[None]
        total_nodes[None] += 1

        if parent_idx != -1:
            if is_second_child:
                nodes[parent_idx].child_1 = current_node_idx
            else:
                nodes[parent_idx].child_0 = current_node_idx

        bounds = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))

        for i in range(start, end):
            bounds = union(bounds, bvh_primitives[i].bounds)

        if bounds.get_surface_area() == 0 or (end - start) == 1:
            first_prim_offset = ti.atomic_add(ordered_prims_idx[None], end - start)
            for i in range(start, end):
                prim_num = bvh_primitives[i].prim_num
                ordered_prims[first_prim_offset + i - start] = primitives[prim_num]
            nodes[current_node_idx].init_leaf(first_prim_offset, end - start, bounds)

        else:
            centroid_bounds = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
            for i in range(start, end):
                centroid_bounds = union_p(centroid_bounds, bvh_primitives[i].bounds.centroid)

            dim = centroid_bounds.get_largest_dim()

            if centroid_bounds.max_point[dim] == centroid_bounds.min_point[dim]:
                first_prim_offset = ti.atomic_add(ordered_prims_idx[None], end - start)
                for i in range(start, end):
                    prim_num = bvh_primitives[i].prim_num
                    ordered_prims[first_prim_offset + i - start] = primitives[prim_num]
                nodes[current_node_idx].init_leaf(first_prim_offset, end - start, bounds)
            else:
                mid = (start + end) // 2  # Default split
                if split_method == 2:
                    mid = partition_equal_counts(bvh_primitives, start, end, dim)
                elif split_method == 1:
                    mid = partition_middle(bvh_primitives, start, end, dim, centroid_bounds)
                elif split_method == 0:
                    if (end - start) <= 2:
                        mid = partition_equal_counts(bvh_primitives, start, end, dim)
                    else:
                        mid = partition_sah(bvh_primitives, start, end, dim, centroid_bounds, costs, buckets, bounds, max_prims_in_node)


                nodes[current_node_idx].split_axis = dim
                nodes[current_node_idx].n_primitives = 0
                nodes[current_node_idx].bounds = bounds

                push(stack, stack_ptr, mid, end, current_node_idx, 1)
                push(stack, stack_ptr, start, mid, current_node_idx, 0)

    return 0



@ti.kernel
def flatten_bvh(
    nodes: ti.template(),
    linear_bvh: ti.template(),
    root: ti.i32,
    stack: ti.template(),
    stack_top: ti.template()
) -> ti.i32:
    offset = ti.i32(0)
    stack_top[None] = 1
    stack[0] = ti.Vector([root, -1, 0])  # (node_idx, parent_idx, is_second_child)

    while stack_top[None] > 0:
        stack_top[None] -= 1
        node_idx, parent_idx, is_second_child = stack[stack_top[None]]

        if node_idx == -1:
            continue

        current_idx = offset
        linear_bvh[current_idx].bounds = nodes[node_idx].bounds
        offset += 1

        if parent_idx != -1 and is_second_child:
            linear_bvh[parent_idx].second_child_offset = current_idx

        if nodes[node_idx].n_primitives > 0:
            # Leaf node
            linear_bvh[current_idx].primitives_offset = nodes[node_idx].first_prim_offset
            linear_bvh[current_idx].n_primitives = nodes[node_idx].n_primitives
        else:
            # Interior node
            linear_bvh[current_idx].axis = nodes[node_idx].split_axis
            linear_bvh[current_idx].n_primitives = 0
            linear_bvh[current_idx].primitives_offset = 0

            if nodes[node_idx].child_1 != -1:
                stack[stack_top[None]] = ti.Vector([nodes[node_idx].child_1, current_idx, 1])
                stack_top[None] += 1

            if nodes[node_idx].child_0 != -1:
                stack[stack_top[None]] = ti.Vector([nodes[node_idx].child_0, current_idx, 0])
                stack_top[None] += 1

    return offset


@ti.func
def intersect_bvh(ray, primitives, nodes, t_min=0.0, t_max=INF):
    intersection = Intersection()

    # Assuming MAX_DEPTH is defined globally or passed as an argument
    nodes_to_visit = ti.Vector([0] * MAX_DEPTH)
    to_visit_offset = 0
    current_node_index = 0
    tMax = t_max
    tMin = t_min

    inv_dir = ti.Vector([1.0 / ray.direction[0], 1.0 / ray.direction[1], 1.0 / ray.direction[2]])
    dir_is_neg = ti.Vector([inv_dir[0] < 0, inv_dir[1] < 0, inv_dir[2] < 0])

    while True:
        node = nodes[current_node_index]
        if intersect_bounds(node.bounds, ray, inv_dir):
            if node.n_primitives > 0:
                # Intersect ray with primitives in leaf BVH node
                for i in range(node.n_primitives):
                    prim_index = node.primitives_offset + i
                    hit, t = primitives[prim_index].intersect(ray.origin, ray.direction, tMax)
                    if hit and tMin < t < tMax:
                        tMax = t
                        intersection.set_intersection(ray, primitives[prim_index], t)
                if to_visit_offset == 0:
                    break
                to_visit_offset -= 1
                current_node_index = nodes_to_visit[to_visit_offset]
            else:
                # Choose which node to visit first based on ray direction
                if dir_is_neg[node.axis]:
                    if to_visit_offset < MAX_DEPTH:
                        nodes_to_visit[to_visit_offset] = current_node_index + 1
                        to_visit_offset += 1
                    current_node_index = node.second_child_offset
                else:
                    if to_visit_offset < MAX_DEPTH:
                        nodes_to_visit[to_visit_offset] = node.second_child_offset
                        to_visit_offset += 1
                    current_node_index = current_node_index + 1
        else:
            if to_visit_offset == 0:
                break
            to_visit_offset -= 1
            current_node_index = nodes_to_visit[to_visit_offset]

    return intersection


@ti.func
def unoccluded(isec_p, isec_n, target_p, primitives, bvh, shadow_epsilon=0.0001):
    direction = normalize(target_p - isec_p)
    distance = (target_p - isec_p).norm() * (1-shadow_epsilon)

    ray = spawn_ray(isec_p, isec_n, direction)

    intersection = intersect_bvh(ray, primitives, bvh, 0, distance)

    return intersection.intersected == 0


@ti.func
def partition_equal_counts(bvh_primitives, start, end, dim):
    mid = (start + end) // 2
    # Sort or partition the primitives based on their centroids along the chosen dimension
    for i in range(start, end):
        min_idx = i
        for j in range(i + 1, end):
            if bvh_primitives[j].bounds.centroid[dim] < bvh_primitives[min_idx].bounds.centroid[dim]:
                min_idx = j
        if min_idx != i:
            bvh_primitives[i], bvh_primitives[min_idx] = bvh_primitives[min_idx], bvh_primitives[i]
    return mid


@ti.func
def partition_middle(bvh_primitives, start, end, dim, centroid_bounds):
    pmid = (centroid_bounds.min_point[dim] + centroid_bounds.max_point[dim]) / 2
    left, right = start, end - 1

    while left <= right:
        while left <= right and bvh_primitives[left].bounds.centroid[dim] < pmid:
            left += 1
        while left <= right and bvh_primitives[right].bounds.centroid[dim] >= pmid:
            right -= 1
        if left < right:
            bvh_primitives[left], bvh_primitives[right] = bvh_primitives[right], bvh_primitives[left]
            left += 1
            right -= 1

    mid = left
    if mid == start or mid == end:
        mid = (start + end) // 2

    return mid


@ti.func
def partition_sah(bvh_primitives, start, end, dim, centroid_bounds, costs, buckets, bounds, max_prims_in_node):
    nBuckets = 12
    nSplits = nBuckets - 1
    minCostSplitBucket = -1
    minCost = INF
    leafCost = (end - start)

    # Initialize buckets
    for b in range(nBuckets):
        buckets[b].count = 0
        buckets[b].bounds = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))

    # Fill buckets
    for i in range(start, end):
        centroid = bvh_primitives[i].bounds.centroid
        b = int(nBuckets * (centroid[dim] - centroid_bounds.min_point[dim]) /
                (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]))
        if b == nBuckets:
            b = nBuckets - 1
        buckets[b].count += 1
        buckets[b].bounds = union(buckets[b].bounds, bvh_primitives[i].bounds)

    # Calculate costs for splits
    countBelow, countAbove = 0, 0
    boundBelow = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
    boundAbove = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))

    # Forward scan for below costs
    for j in range(nSplits):  # Use a unique variable name 'j'
        boundBelow = union(boundBelow, buckets[j].bounds)
        countBelow += buckets[j].count
        costs[j] = countBelow * boundBelow.get_surface_area()

    # Backward scan for above costs
    k = nSplits - 1  # Use a unique variable name 'k'
    while k >= 0:
        boundAbove = union(boundAbove, buckets[k + 1].bounds)
        countAbove += buckets[k + 1].count
        costs[k] += countAbove * boundAbove.get_surface_area()
        k -= 1

    # Find best bucket to split
    for m in range(nSplits):  # Use a unique variable name 'm'
        if costs[m] < minCost:
            minCost = costs[m]
            minCostSplitBucket = m

    minCost = 1.0 / bounds.get_surface_area() * minCost

    # Determine if we should split
    mid = start
    if (end - start) > max_prims_in_node or minCost < leafCost:
        mid = start
        for n in range(start, end):  # Use a unique variable name 'n'
            centroid = bvh_primitives[n].bounds.centroid
            b = int(nBuckets * (centroid[dim] - centroid_bounds.min_point[dim]) /
                    (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]))
            if b == nBuckets:
                b = nBuckets - 1
            if b <= minCostSplitBucket:
                bvh_primitives[mid], bvh_primitives[n] = bvh_primitives[n], bvh_primitives[mid]
                mid += 1
    else:
        mid = start
    return mid
