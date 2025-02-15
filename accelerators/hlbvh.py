import taichi as ti
from taichi.math import vec3
from primitives.aabb import AABB, union, union_p
from accelerators.bvh import BVHNode, BucketInfo  # BVHNode must have methods init_leaf and init_interior
from utils.constants import INF

#--------------------------------------------------------------------
# Data structures
#--------------------------------------------------------------------
@ti.dataclass
class MortonPrimitive:
    prim_idx: ti.i32
    morton_code: ti.i32

@ti.dataclass
class LBVHTreelet:
    start_idx: ti.i32
    n_primitives: ti.i32
    # We assume that for each treelet we will “use” the global node pool,
    # so here we just keep a placeholder (the original Numba code allocated a new list).
    # In Taichi we use the global build_nodes field.
    # (This field is not used directly in our Taichi version.)
    build_nodes: BVHNode

#--------------------------------------------------------------------
# Morton code helpers
#--------------------------------------------------------------------
@ti.func
def left_shift_3(x: ti.i32) -> ti.i32:
    # Use unsigned arithmetic for bit-level operations.
    if x == (1 << 10):
        x = x - 1
    ux = ti.uint32(x)
    ux = (ux | (ux << 16)) & 0x030000FF  # 0b00000011_00000000_00000000_11111111
    ux = (ux | (ux << 8))  & 0x0300F00F
    ux = (ux | (ux << 4))  & 0x030C30C3
    ux = (ux | (ux << 2))  & 0x09249249
    return ti.cast(ux, ti.i32)

@ti.func
def encode_morton_3(v: vec3) -> ti.i32:
    # Assume the vector components are nonnegative.
    return (left_shift_3(int(v[2])) << 2) | (left_shift_3(int(v[1])) << 1) | left_shift_3(int(v[0]))

#--------------------------------------------------------------------
# Radix sort for Morton primitives
# (The implementation is similar to before.)
#--------------------------------------------------------------------
@ti.func
def radix_sort(src: ti.template(), temp: ti.template(),
               bucket_count: ti.template(), bucket_start: ti.template(), n: ti.i32):
    bits_per_pass = 6
    n_bits = 30
    n_passes = n_bits // bits_per_pass
    n_buckets = 1 << bits_per_pass

    for p in range(n_passes):
        low_bit = p * bits_per_pass
        # Reset bucket_count.
        for i in range(n_buckets):
            bucket_count[i] = 0

        if p % 2 == 0:
            for i in range(n):
                bucket = (src[i][0] >> low_bit) & (n_buckets - 1)
                bucket_count[bucket] += 1
            bucket_start[0] = 0
            for i in range(1, n_buckets):
                bucket_start[i] = bucket_start[i - 1] + bucket_count[i - 1]
            for i in range(n):
                bucket = (src[i][0] >> low_bit) & (n_buckets - 1)
                idx = bucket_start[bucket]
                temp[idx] = src[i]
                bucket_start[bucket] = idx + 1
        else:
            for i in range(n):
                bucket = (temp[i][0] >> low_bit) & (n_buckets - 1)
                bucket_count[bucket] += 1
            bucket_start[0] = 0
            for i in range(1, n_buckets):
                bucket_start[i] = bucket_start[i - 1] + bucket_count[i - 1]
            for i in range(n):
                bucket = (temp[i][0] >> low_bit) & (n_buckets - 1)
                idx = bucket_start[bucket]
                src[idx] = temp[i]
                bucket_start[bucket] = idx + 1

    if n_passes % 2 == 1:
        for i in range(n):
            src[i] = temp[i]

#--------------------------------------------------------------------
# Recursive build_upper_sah
# Merges an array of LBVH treelet roots (stored as BVHNodes) into a single BVH.
#--------------------------------------------------------------------
@ti.func
def build_upper_sah(treelet_roots: ti.template(), start: ti.i32, end: ti.i32,
                    total_nodes: ti.template()) -> BVHNode:
    n_nodes = end - start
    if n_nodes == 1:
        return treelet_roots[start]
    total_nodes[None] += 1
    node = BVHNode()  # interior node

    # Compute union of bounds.
    bounds = AABB(vec3(INF, INF, INF), vec3(-INF, -INF, -INF))
    for i in range(start, end):
        bounds = union(bounds, treelet_roots[i].bounds)

    # Compute centroid bounds.
    centroid_bounds = AABB(vec3(INF, INF, INF), vec3(-INF, -INF, -INF))
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
        centroid_bounds = union_p(centroid_bounds, centroid)

    # Choose split dimension.
    dim = 0
    max_extent = -INF
    for i in ti.static(range(3)):
        extent = centroid_bounds.max_point[i] - centroid_bounds.min_point[i]
        if extent > max_extent:
            max_extent = extent
            dim = i

    # Set up buckets.
    n_buckets = 12
    bucket_counts = ti.field(dtype=ti.i32, shape=n_buckets)
    bucket_bounds_min = ti.Vector.field(3, dtype=ti.f32, shape=n_buckets)
    bucket_bounds_max = ti.Vector.field(3, dtype=ti.f32, shape=n_buckets)
    for b in range(n_buckets):
        bucket_counts[b] = 0
        bucket_bounds_min[b] = vec3(INF, INF, INF)
        bucket_bounds_max[b] = vec3(-INF, -INF, -INF)

    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
        ratio = (centroid[dim] - centroid_bounds.min_point[dim]) / (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim] + 1e-5)
        b = ti.min(ti.max(ti.cast(ratio * n_buckets, ti.i32), 0), n_buckets - 1)
        bucket_counts[b] += 1
        bucket_bounds_min[b] = ti.min(bucket_bounds_min[b], treelet_roots[i].bounds.min_point)
        bucket_bounds_max[b] = ti.max(bucket_bounds_max[b], treelet_roots[i].bounds.max_point)

    costs = ti.field(dtype=ti.f32, shape=n_buckets - 1)
    for i in range(n_buckets - 1):
        count0 = 0
        count1 = 0
        bounds0 = AABB(vec3(INF, INF, INF), vec3(-INF, -INF, -INF))
        bounds1 = AABB(vec3(INF, INF, INF), vec3(-INF, -INF, -INF))
        for b in range(0, i + 1):
            count0 += bucket_counts[b]
            bounds0 = union(bounds0, AABB(bucket_bounds_min[b], bucket_bounds_max[b]))
        for b in range(i + 1, n_buckets):
            count1 += bucket_counts[b]
            bounds1 = union(bounds1, AABB(bucket_bounds_min[b], bucket_bounds_max[b]))
        costs[i] = 0.125 + (count0 * bounds0.get_surface_area() + count1 * bounds1.get_surface_area()) / bounds.get_surface_area()
    min_cost = costs[0]
    min_cost_split_bucket = 0
    for i in range(1, n_buckets - 1):
        if costs[i] < min_cost:
            min_cost = costs[i]
            min_cost_split_bucket = i

    mid = start
    for i in range(start, end):
        centroid = (treelet_roots[i].bounds.min_point + treelet_roots[i].bounds.max_point) * 0.5
        ratio = (centroid[dim] - centroid_bounds.min_point[dim]) / (centroid_bounds.max_point[dim] - centroid_bounds.min_point[dim] + 1e-5)
        b = ti.min(ti.max(ti.cast(ratio * n_buckets, ti.i32), 0), n_buckets - 1)
        if b <= min_cost_split_bucket:
            # swap in-place
            temp = treelet_roots[i]
            treelet_roots[i] = treelet_roots[mid]
            treelet_roots[mid] = temp
            mid += 1
    if mid == start or mid == end:
        mid = (start + end) // 2

    left_child = build_upper_sah(treelet_roots, start, mid, total_nodes)
    right_child = build_upper_sah(treelet_roots, mid, end, total_nodes)
    node.init_interior(dim, left_child, right_child, union(left_child.bounds, right_child.bounds))
    return node

#--------------------------------------------------------------------
# Recursive emit_lbvh
#
# This function builds an LBVH treelet from a portion of the sorted Morton
# primitives. We add a "morton_offset" parameter so that we can work on a
# subarray (i.e. a treelet) within the global morton_prims field.
#--------------------------------------------------------------------
@ti.func
def emit_lbvh(build_nodes: ti.template(), primitives: ti.template(), bounded_boxes: ti.template(),
              morton_prims: ti.template(), morton_offset: ti.i32, n_prims: ti.i32,
              total_nodes: ti.template(), ordered_prims: ti.template(),
              ordered_prims_offset: ti.template(), bit_index: ti.i32) -> BVHNode:
    node_ret = BVHNode()  # local variable to hold the return value

    n_boxes = n_prims  # assume one box per primitive
    max_prims_in_node = ti.max(4, ti.cast(0.1 * n_boxes, ti.i32))

    # Base condition: if bit_index is exhausted or few enough primitives, make a leaf.
    if bit_index == -1 or n_prims <= max_prims_in_node:
        total_nodes[None] += 1
        node_ret = build_nodes[total_nodes[None] - 1]  # get next available node
        bounds = AABB(vec3(INF, INF, INF), vec3(-INF, -INF, -INF))
        first_prim_offset = ordered_prims_offset[None]
        ordered_prims_offset[None] += n_prims
        for i in range(n_prims):
            prim_index = morton_prims[morton_offset + i][1]
            ordered_prims[first_prim_offset + i] = primitives[prim_index]
            bounds = union(bounds, bounded_boxes[prim_index].bounds)
        node_ret.init_leaf(first_prim_offset, n_prims, bounds)
    else:
        mask = 1 << bit_index
        # If all primitives share the same bit at the current bit_index...
        if (morton_prims[morton_offset][0] & mask) == (morton_prims[morton_offset + n_prims - 1][0] & mask):
            if bit_index < 2:
                total_nodes[None] += 1
                node_ret = build_nodes[total_nodes[None] - 1]
                bounds = AABB(vec3(INF, INF, INF), vec3(-INF, -INF, -INF))
                first_prim_offset = ordered_prims_offset[None]
                ordered_prims_offset[None] += n_prims
                for i in range(n_prims):
                    prim_index = morton_prims[morton_offset + i][1]
                    ordered_prims[first_prim_offset + i] = primitives[prim_index]
                    bounds = union(bounds, bounded_boxes[prim_index].bounds)
                node_ret.init_leaf(first_prim_offset, n_prims, bounds)
            else:
                # Instead of recursing with the same partition, reduce bit_index.
                node_ret = emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims,
                                     morton_offset, n_prims, total_nodes, ordered_prims,
                                     ordered_prims_offset, bit_index - 1)
        else:
            # Find the split position.
            split_offset = 0
            for i in range(n_prims - 1):
                if (morton_prims[morton_offset + i][0] & mask) != (morton_prims[morton_offset + i + 1][0] & mask):
                    split_offset = i + 1
                    break
            # If no split is found, force a split by taking the middle.
            if split_offset == 0:
                split_offset = n_prims // 2

            total_nodes[None] += 1
            node_ret = build_nodes[total_nodes[None] - 1]
            left_node = emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims,
                                  morton_offset, split_offset, total_nodes, ordered_prims,
                                  ordered_prims_offset, bit_index - 1)
            right_node = emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims,
                                   morton_offset + split_offset, n_prims - split_offset, total_nodes,
                                   ordered_prims, ordered_prims_offset, bit_index - 1)
            axis = bit_index % 3
            node_ret.init_interior(axis, left_node, right_node, union(left_node.bounds, right_node.bounds))

    return node_ret


#--------------------------------------------------------------------
# Top-level LBVH builder kernel
#
# This kernel performs the following steps:
#  (1) Compute the global centroid bounds and Morton codes.
#  (2) Radix sort the Morton primitives.
#  (3) Partition the sorted Morton primitives into treelets (using a mask).
#  (4) For each treelet, call emit_lbvh to build its LBVH.
#  (5) Merge the finished treelets via build_upper_sah.
#  (6) Finally, store the final BVH root into build_nodes[0].
#--------------------------------------------------------------------
@ti.kernel
def build_hlbvh(primitives: ti.template(), bounded_boxes: ti.template(),
                morton_prims: ti.template(), n_prims: ti.i32,
                ordered_prims: ti.template(), build_nodes: ti.template(),
                total_nodes: ti.template(), temp: ti.template(),
                bucket_count: ti.template(), bucket_start: ti.template(),
                ordered_prims_offset: ti.template(),
                treelet_start: ti.template(), treelet_n: ti.template(),
                num_treelets: ti.template(), finished_treelets: ti.template(),
                nodes_array: ti.template()):
    # (1) Compute global centroid bounds.
    global_min = vec3(INF, INF, INF)
    global_max = vec3(-INF, -INF, -INF)
    for i in range(n_prims):
        c = bounded_boxes[i].bounds.centroid
        global_min = ti.min(global_min, c)
        global_max = ti.max(global_max, c)
    global_bounds = AABB(global_min, global_max)
    mortonScale = 1 << 10

    # (2) Compute Morton codes.
    for i in range(n_prims):
        c = bounded_boxes[i].bounds.centroid
        off = (c - global_bounds.min_point) / (global_bounds.max_point - global_bounds.min_point + 1e-5)
        scaled = vec3(int(off[0] * mortonScale),
                      int(off[1] * mortonScale),
                      int(off[2] * mortonScale))
        code = encode_morton_3(scaled)
        morton_prims[i][0] = code
        morton_prims[i][1] = bounded_boxes[i].prim_num

    # (3) Radix sort the Morton primitives.
    radix_sort(morton_prims, temp, bucket_count, bucket_start, n_prims)

    # (4) Partition the sorted morton_prims into treelets.
    mask = 0b00111111111111000000000000000000
    num_treelets[None] = 0
    var_start = 0
    for end in range(1, n_prims + 1):
        if end == n_prims or ((morton_prims[var_start][0] & mask) != (morton_prims[end][0] & mask)):
            treelet_start[num_treelets[None]] = var_start
            treelet_n[num_treelets[None]] = end - var_start
            num_treelets[None] += 1
            var_start = end

    # (5) For each treelet, call emit_lbvh.
    first_bit_index = 29 - 12
    for i in range(num_treelets[None]):
        tr_start = treelet_start[i]
        tr_n = treelet_n[i]
        tr_root = emit_lbvh(build_nodes, primitives, bounded_boxes, morton_prims, tr_start, tr_n,
                             total_nodes, ordered_prims, ordered_prims_offset, first_bit_index)
        finished_treelets[i][0] = tr_root
        nodes_array[i] = 0  # For debugging or statistics.

    # (6) Merge all finished treelets via build_upper_sah.
    final_bvh = build_upper_sah(finished_treelets, 0, num_treelets[None], total_nodes)
    build_nodes[0] = final_bvh
