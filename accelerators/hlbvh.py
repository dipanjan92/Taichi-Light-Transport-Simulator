import taichi as ti
from taichi.math import vec3

from accelerators.bvh import BVHNode
from primitives.aabb import AABB, union_p, union
from utils.constants import INF


@ti.dataclass
class MortonPrimitive:
    prim_idx: ti.i32
    morton_code: ti.i32


@ti.dataclass
class LBVHTreelet:
    start_idx: ti.i32
    n_primitives: ti.i32
    build_nodes: BVHNode


@ti.func
def left_shift_3(x):
    # Assuming x is a 10-bit value
    assert x <= (1 << 10), "x is bigger than 2^10"
    if x == (1 << 10):
        x = x - 1

    x = (x | (x << 16)) & 0b00000011000000000000000011111111
    x = (x | (x << 8)) & 0b00000011000000001111000000001111
    x = (x | (x << 4)) & 0b00000011000011000011000011000011
    x = (x | (x << 2)) & 0b00001001001001001001001001001001

    return x

@ti.func
def encode_morton_3(v):
    assert v[0] >= 0, "x must be non-negative"
    assert v[1] >= 0, "y must be non-negative"
    assert v[2] >= 0, "z must be non-negative"
    return (left_shift_3(v[2]) << 2) | (left_shift_3(v[1]) << 1) | left_shift_3(v[0])


@ti.kernel
def radix_sort(v: ti.template()):
    n_primitives = v.shape[0]
    temp_vector = ti.Vector.field(2, dtype=ti.i32, shape=n_primitives)  # MortonPrimitive replacement
    bits_per_pass = 6
    n_bits = 30
    n_passes = n_bits // bits_per_pass
    n_buckets = 1 << bits_per_pass

    for _pass in range(n_passes):
        low_bit = _pass * bits_per_pass

        # Set in and out vector pointers for radix sort pass
        in_v = v if _pass % 2 == 0 else temp_vector
        out_v = temp_vector if _pass % 2 == 0 else v

        bucket_count = ti.field(dtype=ti.i32, shape=n_buckets)
        out_ix = ti.field(dtype=ti.i32, shape=n_buckets)

        # Count the number of zero bits
        for i in range(n_primitives):
            bucket = (in_v[i][1] >> low_bit) & (n_buckets - 1)
            bucket_count[bucket] += 1

        # Compute starting index in output array for each bucket
        for i in range(1, n_buckets):
            out_ix[i] = out_ix[i - 1] + bucket_count[i - 1]

        # Store sorted values in output array
        for i in range(n_primitives):
            bucket = (in_v[i][1] >> low_bit) & (n_buckets - 1)
            out_v[out_ix[bucket]] = in_v[i]
            out_ix[bucket] += 1

    # If the last pass ended with temp_vector, we need to copy it back
    if n_passes % 2 == 1:
        for i in range(n_primitives):
            v[i] = temp_vector[i]


@ti.func
def push_sah(stack, stack_ptr, start, end, node_idx, is_second_child):
    stack[stack_ptr[None], 0] = start
    stack[stack_ptr[None], 1] = end
    stack[stack_ptr[None], 2] = node_idx
    stack[stack_ptr[None], 3] = is_second_child
    stack_ptr[None] += 1


@ti.func
def pop_sah(stack, stack_ptr):
    stack_ptr[None] -= 1
    start = stack[stack_ptr[None], 0]
    end = stack[stack_ptr[None], 1]
    node_idx = stack[stack_ptr[None], 2]
    is_second_child = stack[stack_ptr[None], 3]
    return start, end, node_idx, is_second_child


@ti.kernel
def build_upper_sah(treelet_roots: ti.template(), total_nodes: ti.template()) -> ti.i32:
    push_sah(stack, stack_ptr, 0, treelet_roots.shape[0], -1, 0)

    while stack_ptr[None] > 0:
        start, end, parent_node_idx, is_second_child = pop_sah(stack, stack_ptr)
        nNodes = end - start

        if nNodes == 1:
            current_node_idx = start
        else:
            total_nodes[None] += 1
            current_node_idx = total_nodes[None] - 1
            node = BVHNode()

            # Compute bounds of all nodes under this HLBVH node
            total_bounds = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
            for i in range(start, end):
                total_bounds = union_bounds(total_bounds, treelet_roots[i].bounds)

            # Compute bound of HLBVH node centroids, choose split dimension `dim`
            centroid_bounds = compute_centroid_bounds(treelet_roots, start, end)
            dim = centroid_bounds.get_largest_dim()

            if centroid_bounds.max_point[dim] == centroid_bounds.min_point[dim]:
                current_node_idx = start
            else:
                # Partition treelet roots into buckets
                buckets, counts = compute_partition_buckets(treelet_roots, start, end, dim, centroid_bounds)

                # Compute costs for splitting after each bucket
                costs = compute_split_costs(buckets, counts, len(buckets), total_bounds)

                # Find best split bucket
                min_cost_split_bucket = find_best_split(costs, len(buckets))

                # Partition treelet roots at the chosen split bucket
                mid = start
                for i in range(start, end):
                    centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
                    b = int(len(buckets) * (centroid[dim] - centroid_bounds.min_point[dim]) /
                            (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim]))
                    if b == len(buckets):
                        b = len(buckets) - 1
                    if b <= min_cost_split_bucket:
                        treelet_roots[mid], treelet_roots[i] = treelet_roots[i], treelet_roots[mid]
                        mid += 1

                # Push child nodes to the stack for further processing
                push_sah(stack, stack_ptr, mid, end, current_node_idx, 1)
                push_sah(stack, stack_ptr, start, mid, current_node_idx, 0)

            node.bounds = total_bounds

        if parent_node_idx != -1:
            if is_second_child:
                treelet_roots[parent_node_idx].child_1 = current_node_idx
            else:
                treelet_roots[parent_node_idx].child_0 = current_node_idx

    return treelet_roots[0]


@ti.kernel
def build_hlbvh(bvhPrimitives: ti.template(),
                mortonPrims: ti.template(),
                orderedPrims: ti.template(),
                totalNodes: ti.template(),
                orderedPrimsOffset: ti.template(),
                buildNodes: ti.template(),
                finishedTreelets: ti.template(),
                max_prims_in_node: ti.i32) -> ti.i32:
    bounds = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))

    # Compute bounding box of all primitive centroids
    for i in range(bvhPrimitives.shape[0]):
        bounds = union(bounds, bvhPrimitives[i].centroid())

    # Compute Morton indices of primitives
    mortonBits = 10
    mortonScale = 1 << mortonBits

    for i in range(nPrimitives):
        centroidOffset = bounds.offset(bvhPrimitives[i].centroid())
        offset = centroidOffset * mortonScale
        mortonCode = encode_morton3(offset.x, offset.y, offset.z)  # Implement this function
        mortonPrims[i][0] = mortonCode
        mortonPrims[i][1] = i  # Storing the primitive index

    # Sort primitive Morton indices
    radix_sort(mortonPrims)  # Implement this function

    # Create LBVH treelets at bottom of BVH
    treeletsToBuild = []
    start, end = 0, 1

    while end <= nPrimitives:
        mask = 0b00111111111111000000000000000000
        if end == nPrimitives or (mortonPrims[start][0] & mask) != (mortonPrims[end][0] & mask):
            nTreeletPrimitives = end - start
            maxBVHNodes = 2 * nTreeletPrimitives - 1
            treeletsToBuild.append((start, end, maxBVHNodes))
            start = end
        end += 1

    # Create LBVHs for treelets
    for i in range(len(treeletsToBuild)):
        startIndex, endIndex, maxBVHNodes = treeletsToBuild[i]
        nodesCreated = 0
        emit_lbvh(buildNodes, bvhPrimitives, mortonPrims, startIndex, endIndex, totalNodes, orderedPrims,
                  orderedPrimsOffset, max_prims_in_node)

    # Create and return SAH BVH from LBVH treelets
    for i in range(len(treeletsToBuild)):
        finishedTreelets[i] = buildNodes[i]

    return build_upper_sah(finishedTreelets, 0, len(treeletsToBuild), totalNodes)  # Implement this function


@ti.func
def emit_lbvh(buildNodes: ti.template(),
              bvhPrimitives: ti.template(),
              mortonPrims: ti.template(),
              startIndex: ti.i32,
              endIndex: ti.i32,
              totalNodes: ti.template(),
              orderedPrims: ti.template(),
              orderedPrimsOffset: ti.template(),
              max_prims_in_node: ti.i32) -> ti.i32:
    push_lbvh(stack, stack_ptr, startIndex, endIndex, 29 - 12, -1)

    while stack_ptr[None] > 0:
        start, end, bitIndex, parentNodeIdx = pop_lbvh(stack, stack_ptr)
        nCurrentPrimitives = end - start

        if nCurrentPrimitives <= max_prims_in_node or bitIndex == -1:
            totalNodes[None] += 1
            currentNodeIdx = buildNodes[None]
            buildNodes[None] += 1

            bounds = AABB(vec3([INF] * 3), vec3([-INF] * 3), vec3([INF] * 3))
            firstPrimOffset = ti.atomic_add(orderedPrimsOffset[None], nCurrentPrimitives)

            for i in range(start, end):
                primitiveIndex = mortonPrims[i][1]
                orderedPrims[firstPrimOffset + i - start] = primitiveIndex
                bounds = union(bounds, bvhPrimitives[primitiveIndex].bounds)

            if parentNodeIdx != -1:
                if parentNodeIdx < currentNodeIdx:
                    buildNodes[parentNodeIdx].child_1 = currentNodeIdx
                else:
                    buildNodes[parentNodeIdx].child_0 = currentNodeIdx

            buildNodes[currentNodeIdx].init_leaf(firstPrimOffset, nCurrentPrimitives, bounds)

        else:
            mask = 1 << bitIndex
            splitOffset = start

            for i in range(start + 1, end):
                if (mortonPrims[i - 1][0] & mask) != (mortonPrims[i][0] & mask):
                    splitOffset = i
                    break

            totalNodes[None] += 1
            currentNodeIdx = buildNodes[None]
            buildNodes[None] += 1

            if parentNodeIdx != -1:
                if parentNodeIdx < currentNodeIdx:
                    buildNodes[parentNodeIdx].child_1 = currentNodeIdx
                else:
                    buildNodes[parentNodeIdx].child_0 = currentNodeIdx

            axis = bitIndex % 3
            buildNodes[currentNodeIdx].split_axis = axis
            buildNodes[currentNodeIdx].n_primitives = 0

            push_lbvh(stack, stack_ptr, splitOffset, end, bitIndex - 1, currentNodeIdx)
            push_lbvh(stack, stack_ptr, start, splitOffset, bitIndex - 1, currentNodeIdx)


@ti.func
def push_lbvh(stack, stack_ptr, start, end, bitIndex, nodeIdx):
    stack[stack_ptr[None], 0] = start
    stack[stack_ptr[None], 1] = end
    stack[stack_ptr[None], 2] = bitIndex
    stack[stack_ptr[None], 3] = nodeIdx
    stack_ptr[None] += 1


@ti.func
def pop_lbvh(stack, stack_ptr):
    stack_ptr[None] -= 1
    start = stack[stack_ptr[None], 0]
    end = stack[stack_ptr[None], 1]
    bitIndex = stack[stack_ptr[None], 2]
    nodeIdx = stack[stack_ptr[None], 3]
    return start, end, bitIndex, nodeIdx