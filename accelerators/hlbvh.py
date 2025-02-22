import taichi as ti
from taichi.math import vec3

# ---------------------------------------------------------------------
# Imports from your existing code (AABB, BVHNode, etc.)
# ---------------------------------------------------------------------
from primitives.aabb import AABB, union, union_p, BVHPrimitive
from accelerators.bvh import BVHNode
from utils.constants import INF


@ti.dataclass
class MortonPrimitive:
    prim_idx: ti.i32
    morton_code: ti.u32

@ti.dataclass
class LBVHTreelet:
    start_index: ti.i32
    n_primitives: ti.i32
    build_nodes_start: ti.i32  # index into BVHNode array


MAX_PRIMS = 1_000_000

morton_prims      = MortonPrimitive.field(shape=(MAX_PRIMS,))
morton_prims_temp = MortonPrimitive.field(shape=(MAX_PRIMS,))
treelets          = LBVHTreelet.field(shape=(MAX_PRIMS,))

n_morton    = ti.field(ti.i32, shape=())
n_treelets  = ti.field(ti.i32, shape=())
treelet_roots = ti.field(ti.i32, shape=(MAX_PRIMS,))

centroid_bounds_min = ti.Vector.field(3, dtype=ti.f32, shape=())
centroid_bounds_max = ti.Vector.field(3, dtype=ti.f32, shape=())
lbvh_nodes_next     = ti.field(ti.i32, shape=())

# Arrays for 6-bit Radix Sort:
RADIX_BUCKETS = 64
bucket_count_radix = ti.field(ti.i32, shape=RADIX_BUCKETS)

# Stacks for LBVH emission and upper SAH building:
EMIT_STACK_SIZE = 512
emit_stack = ti.Vector.field(5, dtype=ti.i32, shape=EMIT_STACK_SIZE)
emit_stack_ptr = ti.field(ti.i32, shape=())

UPPER_SAH_STACK_SIZE = 512
upper_sah_stack = ti.Vector.field(4, dtype=ti.i32, shape=UPPER_SAH_STACK_SIZE)
upper_sah_stack_ptr = ti.field(ti.i32, shape=())

# SAH Bucket fields for top-level "build_upper_sah":
NBUCKETS = 12
bucket_count_hlbvh = ti.field(ti.i32, shape=NBUCKETS)
bucket_bmin_hlbvh  = ti.Vector.field(3, dtype=ti.f32, shape=NBUCKETS)
bucket_bmax_hlbvh  = ti.Vector.field(3, dtype=ti.f32, shape=NBUCKETS)
cost_arr_hlbvh     = ti.field(ti.f32, shape=NBUCKETS-1)

# "Local arrays" for build_upper_sah, declared globally
build_upper_sah_root     = ti.field(ti.i32, shape=MAX_PRIMS)
build_upper_sah_minp     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PRIMS)
build_upper_sah_maxp     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PRIMS)
build_upper_sah_centroid = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PRIMS)


@ti.func
def expand_bits_10(v: ti.u32) -> ti.u32:
    v64 = ti.u64(v)
    v64 = (v64 * ti.u64(0x00010001)) & ti.u64(0xFF0000FF)
    v64 = (v64 * ti.u64(0x00000101)) & ti.u64(0x0F00F00F)
    v64 = (v64 * ti.u64(0x00000011)) & ti.u64(0xC30C30C3)
    v64 = (v64 * ti.u64(0x00000005)) & ti.u64(0x49249249)
    return ti.u32(v64)

@ti.func
def encode_morton3(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
    xx = ti.min(ti.max(x, 0.0), 1023.0)
    yy = ti.min(ti.max(y, 0.0), 1023.0)
    zz = ti.min(ti.max(z, 0.0), 1023.0)
    ix = expand_bits_10(ti.u32(xx))
    iy = expand_bits_10(ti.u32(yy))
    iz = expand_bits_10(ti.u32(zz))
    return (ix << 2) | (iy << 1) | iz


@ti.func
def morton_radix_sort(n: ti.i32):
    bits_per_pass = 6
    n_bits = 30
    n_passes = n_bits // bits_per_pass
    bit_mask = (1 << bits_per_pass) - 1
    pass_index = 0
    while pass_index < n_passes:
        low_bit = pass_index * bits_per_pass
        for b in range(RADIX_BUCKETS):
            bucket_count_radix[b] = 0
        for i in range(n):
            mp = morton_prims[i]
            # Cast the bucket index explicitly to int
            bucket = int((mp.morton_code >> low_bit) & ti.u32(bit_mask))
            ti.atomic_add(bucket_count_radix[bucket], 1)
        for b in range(1, RADIX_BUCKETS):
            bucket_count_radix[b] += bucket_count_radix[b - 1]
        for r_i in range(n):
            i = n - 1 - r_i
            mp = morton_prims[i]
            bucket = int((mp.morton_code >> low_bit) & ti.u32(bit_mask))
            bucket_count_radix[bucket] -= 1
            out_idx = bucket_count_radix[bucket]
            morton_prims_temp[out_idx] = mp
        for i in range(n):
            morton_prims[i] = morton_prims_temp[i]
        pass_index += 1

@ti.func
def find_interval(n_prims: ti.i32, start_idx: ti.i32, mask: ti.u32) -> ti.i32:
    i = 1
    base_code = morton_prims[start_idx].morton_code
    base_val  = base_code & mask
    while i < n_prims:
        if (morton_prims[start_idx + i].morton_code & mask) != base_val:
            break
        i += 1
    return i - 1


@ti.func
def check_and_fix_cycle(
    cur_node_idx: ti.i32,
    parent_idx: ti.i32,
    child_idx: ti.i32,
    s_idx: ti.i32,
    s_count: ti.i32,
    bvh_primitives: ti.template(),
    primitives: ti.template(),
    ordered_prims: ti.template(),
    ordered_prims_offset: ti.template(),
    nodes: ti.template()
) -> ti.i32:
    cyc = 0
    if (child_idx == cur_node_idx) or (child_idx == parent_idx):
        cyc = 1
    if cyc == 1:
        # Force current node to become a leaf.
        b = AABB(vec3([INF, INF, INF]),
                 vec3([-INF, -INF, -INF]),
                 vec3([INF, INF, INF]))
        offset_here = ti.atomic_add(ordered_prims_offset[None], s_count)
        for i in range(s_idx, s_idx + s_count):
            prim_id = morton_prims[i].prim_idx
            b = union(b, bvh_primitives[prim_id].bounds)
            ordered_prims[offset_here + (i - s_idx)] = primitives[prim_id]
        nodes[cur_node_idx].init_leaf(offset_here, s_count, b)
    return cyc


@ti.func
def emit_lbvhtreelet(
    start_idx: ti.i32,
    n_prims: ti.i32,
    bit_index: ti.i32,
    bvh_primitives: ti.template(),
    primitives: ti.template(),          # original geometry
    ordered_prims: ti.template(),         # re-ordered geometry array
    total_nodes: ti.template(),
    ordered_prims_offset: ti.template(),
    nodes: ti.template(),
    max_prims_in_node: ti.template(),
    stack: ti.template(),
    stack_ptr: ti.template()
) -> ti.i32:
    stack_ptr[None] = 0
    # Push initial range.
    stack[stack_ptr[None]][0] = start_idx
    stack[stack_ptr[None]][1] = n_prims
    stack[stack_ptr[None]][2] = bit_index
    stack[stack_ptr[None]][3] = -1
    stack[stack_ptr[None]][4] = 0
    stack_ptr[None] += 1

    root_idx = -1
    while stack_ptr[None] > 0:
        stack_ptr[None] -= 1
        s_idx   = stack[stack_ptr[None]][0]
        s_count = stack[stack_ptr[None]][1]
        s_bit   = stack[stack_ptr[None]][2]
        s_parent= stack[stack_ptr[None]][3]
        s_is_sec= stack[stack_ptr[None]][4]

        # If no further bits or few primitives, create a leaf.
        if (s_bit == -1) or (s_count <= max_prims_in_node[None]):
            cur_node_idx = ti.atomic_add(total_nodes[None], 1)
            if s_parent == -1:
                root_idx = cur_node_idx
            else:
                if s_is_sec == 1:
                    nodes[s_parent].child_1 = cur_node_idx
                else:
                    nodes[s_parent].child_0 = cur_node_idx
            b = AABB(vec3([INF, INF, INF]),
                     vec3([-INF, -INF, -INF]),
                     vec3([INF, INF, INF]))
            offset_here = ti.atomic_add(ordered_prims_offset[None], s_count)
            for i in range(s_idx, s_idx + s_count):
                prim_id = morton_prims[i].prim_idx
                b = union(b, bvh_primitives[prim_id].bounds)
                ordered_prims[offset_here + (i - s_idx)] = primitives[prim_id]
            nodes[cur_node_idx].init_leaf(offset_here, s_count, b)
            continue

        # Otherwise, try splitting using the current bit.
        first_code = morton_prims[s_idx].morton_code
        last_code  = morton_prims[s_idx + s_count - 1].morton_code
        mask = 1 << s_bit
        if (first_code & mask) == (last_code & mask):
            stp = stack_ptr[None]
            stack[stp][0] = s_idx
            stack[stp][1] = s_count
            stack[stp][2] = s_bit - 1
            stack[stp][3] = s_parent
            stack[stp][4] = s_is_sec
            stack_ptr[None] += 1
            continue

        interval_offset = find_interval(s_count, s_idx, mask)
        split_offset = interval_offset + 1
        if split_offset > s_count - 1:
            split_offset = s_count - 1

        cur_node_idx = ti.atomic_add(total_nodes[None], 1)
        if s_parent == -1:
            root_idx = cur_node_idx
        else:
            if s_is_sec == 1:
                nodes[s_parent].child_1 = cur_node_idx
            else:
                nodes[s_parent].child_0 = cur_node_idx

        axis = s_bit % 3
        dummy_bounds = AABB(vec3([INF, INF, INF]),
                            vec3([-INF, -INF, -INF]),
                            vec3([INF, INF, INF]))
        nodes[cur_node_idx].init_interior(axis, -1, -1, dummy_bounds)

        # Check for cycles. If a cycle is detected, force a leaf.
        if check_and_fix_cycle(cur_node_idx, s_parent, cur_node_idx,
                               s_idx, s_count, bvh_primitives, primitives,
                               ordered_prims, ordered_prims_offset, nodes) == 1:
            continue

        # Fallback: if no valid split was produced, force this node to be a leaf.
        if (split_offset <= 0) or (split_offset >= s_count):
            b = AABB(vec3([INF, INF, INF]),
                     vec3([-INF, -INF, -INF]),
                     vec3([INF, INF, INF]))
            offset_here = ti.atomic_add(ordered_prims_offset[None], s_count)
            for i in range(s_idx, s_idx + s_count):
                prim_id = morton_prims[i].prim_idx
                b = union(b, bvh_primitives[prim_id].bounds)
                ordered_prims[offset_here + (i - s_idx)] = primitives[prim_id]
            nodes[cur_node_idx].init_leaf(offset_here, s_count, b)
            continue

        # Otherwise, push the two child ranges.
        stp = stack_ptr[None]
        stack[stp][0] = s_idx
        stack[stp][1] = split_offset
        stack[stp][2] = s_bit - 1
        stack[stp][3] = cur_node_idx
        stack[stp][4] = 0
        stack_ptr[None] += 1

        stp = stack_ptr[None]
        stack[stp][0] = s_idx + split_offset
        stack[stp][1] = s_count - split_offset
        stack[stp][2] = s_bit - 1
        stack[stp][3] = cur_node_idx
        stack[stp][4] = 1
        stack_ptr[None] += 1

    return root_idx


@ti.func
def build_upper_sah(
    nodes: ti.template(),
    total_nodes: ti.template(),
    n_trees: ti.i32,
    max_prims_in_node: ti.template(),
    bucket_count: ti.template(),
    bucket_bmin: ti.template(),
    bucket_bmax: ti.template(),
    cost_arr: ti.template(),
    stack: ti.template(),
    stack_ptr: ti.template(),
    local_root: ti.template(),
    local_minp: ti.template(),
    local_maxp: ti.template(),
    local_centroid: ti.template()
) -> ti.i32:
    res = -1
    if n_trees > 0:
        if n_trees == 1:
            res = treelet_roots[0]
        else:
            # Fill local arrays with subtree bounding information.
            for i in range(n_treelets[None]):
                r = treelet_roots[i]
                local_root[i] = r
                local_minp[i] = nodes[r].bounds.min_point
                local_maxp[i] = nodes[r].bounds.max_point
                local_centroid[i] = 0.5 * (nodes[r].bounds.min_point + nodes[r].bounds.max_point)
            # Push entire range [0, n_trees)
            stack_ptr[None] = 0
            stack[0][0] = 0
            stack[0][1] = n_trees
            stack[0][2] = -1
            stack[0][3] = 0
            stack_ptr[None] = 1

            local_root_of_build = -1
            while stack_ptr[None] > 0:
                stack_ptr[None] -= 1
                s = stack[stack_ptr[None]][0]
                e = stack[stack_ptr[None]][1]
                parent = stack[stack_ptr[None]][2]
                is_sec = stack[stack_ptr[None]][3]
                cur_node_idx = ti.atomic_add(total_nodes[None], 1)
                if parent == -1:
                    local_root_of_build = cur_node_idx
                else:
                    if is_sec == 1:
                        nodes[parent].child_1 = cur_node_idx
                    else:
                        nodes[parent].child_0 = cur_node_idx
                # Compute bounding box for subtrees in [s, e)
                bmin = vec3([INF, INF, INF])
                bmax = vec3([-INF, -INF, -INF])
                for i2 in range(s, e):
                    bmin = ti.min(bmin, local_minp[i2])
                    bmax = ti.max(bmax, local_maxp[i2])
                temp_bounds = AABB(bmin, bmax, 0.5 * (bmin + bmax))
                count = e - s
                if count == 1:
                    # For a single subtree, simply pass it through instead of forcing an interior node with a missing child.
                    if parent == -1:
                        local_root_of_build = local_root[s]
                    else:
                        if is_sec == 1:
                            nodes[parent].child_1 = local_root[s]
                        else:
                            nodes[parent].child_0 = local_root[s]
                    continue
                else:
                    # Compute centroid bounds.
                    cmin = vec3([INF, INF, INF])
                    cmax = vec3([-INF, -INF, -INF])
                    for i3 in range(s, e):
                        cmin = ti.min(cmin, local_centroid[i3])
                        cmax = ti.max(cmax, local_centroid[i3])
                    dx = cmax[0] - cmin[0]
                    dy = cmax[1] - cmin[1]
                    dz = cmax[2] - cmin[2]
                    axis = 0
                    if dx > dy and dx > dz:
                        axis = 0
                    elif dy > dz:
                        axis = 1
                    else:
                        axis = 2
                    # Reset buckets.
                    for b in range(NBUCKETS):
                        bucket_count[b] = 0
                        bucket_bmin[b] = vec3([INF, INF, INF])
                        bucket_bmax[b] = vec3([-INF, -INF, -INF])
                    for k in range(NBUCKETS - 1):
                        cost_arr[k] = 0.0
                    cd = cmax[axis] - cmin[axis]
                    for i3 in range(s, e):
                        cval = 0.5 * (local_minp[i3][axis] + local_maxp[i3][axis])
                        rel = (cval - cmin[axis]) / (cd if cd != 0 else 1e-8)
                        if rel < 0:
                            rel = 0
                        if rel >= 1:
                            rel = 0.999999
                        bidx = int(rel * NBUCKETS)
                        if bidx == NBUCKETS:
                            bidx = NBUCKETS - 1
                        bucket_count[bidx] += 1
                        bbmin = bucket_bmin[bidx]
                        bbmax = bucket_bmax[bidx]
                        bbmin = ti.min(bbmin, local_minp[i3])
                        bbmax = ti.max(bbmax, local_maxp[i3])
                        bucket_bmin[bidx] = bbmin
                        bucket_bmax[bidx] = bbmax
                    count_below = 0
                    bmin_below = vec3([INF, INF, INF])
                    bmax_below = vec3([-INF, -INF, -INF])
                    for b in range(NBUCKETS - 1):
                        bmin_below = ti.min(bmin_below, bucket_bmin[b])
                        bmax_below = ti.max(bmax_below, bucket_bmax[b])
                        count_below += bucket_count[b]
                        diag = bmax_below - bmin_below
                        area_below = 2.0 * (diag[0]*diag[1] + diag[0]*diag[2] + diag[1]*diag[2])
                        cost_arr[b] = count_below * area_below
                    count_above = 0
                    bmin_above = vec3([INF, INF, INF])
                    bmax_above = vec3([-INF, -INF, -INF])
                    for rb in range(NBUCKETS):
                        b_ = (NBUCKETS - 1) - rb
                        bmin_above = ti.min(bmin_above, bucket_bmin[b_])
                        bmax_above = ti.max(bmax_above, bucket_bmax[b_])
                        count_above += bucket_count[b_]
                        if b_ > 0:
                            diag = bmax_above - bmin_above
                            area_above = 2.0 * (diag[0]*diag[1] + diag[0]*diag[2] + diag[1]*diag[2])
                            cost_arr[b_-1] += count_above * area_above
                    diag_all = bmax - bmin
                    area_all = 2.0 * (diag_all[0]*diag_all[1] + diag_all[0]*diag_all[2] + diag_all[1]*diag_all[2])
                    leaf_cost = float(count)
                    min_cost = 1e30
                    best_split = 0
                    for kk in range(NBUCKETS - 1):
                        cst = cost_arr[kk] * (1.0/area_all) + 0.125
                        if cst < min_cost:
                            min_cost = cst
                            best_split = kk
                    do_split = 0
                    if (count > max_prims_in_node[None]) and (min_cost < leaf_cost):
                        do_split = 1
                    if do_split == 0:
                        # Fallback to equal-partition (bubble-sort partition by centroid)
                        for i4 in range(s, e):
                            min_idx = i4
                            for j4 in range(i4 + 1, e):
                                if local_centroid[j4][axis] < local_centroid[min_idx][axis]:
                                    min_idx = j4
                            if min_idx != i4:
                                tmp_root = local_root[i4]
                                tmp_min = local_minp[i4]
                                tmp_max = local_maxp[i4]
                                tmp_cent = local_centroid[i4]
                                local_root[i4] = local_root[min_idx]
                                local_minp[i4] = local_minp[min_idx]
                                local_maxp[i4] = local_maxp[min_idx]
                                local_centroid[i4] = local_centroid[min_idx]
                                local_root[min_idx] = tmp_root
                                local_minp[min_idx] = tmp_min
                                local_maxp[min_idx] = tmp_max
                                local_centroid[min_idx] = tmp_cent
                        mid = (s + e) // 2
                        # Modification: if the partition is degenerate, instead of initializing an interior node with a missing child,
                        # simply pass through the only subtree.
                        if (mid <= s) or (mid >= e):
                            if parent == -1:
                                local_root_of_build = local_root[s]
                            else:
                                if is_sec == 1:
                                    nodes[parent].child_1 = local_root[s]
                                else:
                                    nodes[parent].child_0 = local_root[s]
                            continue
                        else:
                            sp = stack_ptr[None]
                            stack[sp][0] = mid
                            stack[sp][1] = e
                            stack[sp][2] = cur_node_idx
                            stack[sp][3] = 1
                            stack_ptr[None] += 1
                            sp = stack_ptr[None]
                            stack[sp][0] = s
                            stack[sp][1] = mid
                            stack[sp][2] = cur_node_idx
                            stack[sp][3] = 0
                            stack_ptr[None] += 1
                            nodes[cur_node_idx].init_interior(axis, -1, -1, temp_bounds)
                    else:
                        # Actual SAH partition.
                        left = s
                        right = e - 1
                        while left <= right:
                            cval = 0.5 * (local_minp[left][axis] + local_maxp[left][axis])
                            rel  = (cval - cmin[axis]) / (cd if cd != 0 else 1e-8)
                            if rel < 0:
                                rel = 0
                            if rel >= 1:
                                rel = 0.999999
                            lb_ = int(rel * NBUCKETS)
                            if lb_ == NBUCKETS:
                                lb_ = NBUCKETS - 1
                            if lb_ <= best_split:
                                left += 1
                            else:
                                tmp_root = local_root[left]
                                tmp_min = local_minp[left]
                                tmp_max = local_maxp[left]
                                tmp_cent = local_centroid[left]
                                local_root[left] = local_root[right]
                                local_minp[left] = local_minp[right]
                                local_maxp[left] = local_maxp[right]
                                local_centroid[left] = local_centroid[right]
                                local_root[right] = tmp_root
                                local_minp[right] = tmp_min
                                local_maxp[right] = tmp_max
                                local_centroid[right] = tmp_cent
                                right -= 1
                        mid = left
                        # Modification: if partition is degenerate, pass through the only subtree.
                        if (mid <= s) or (mid >= e):
                            if parent == -1:
                                local_root_of_build = local_root[s]
                            else:
                                if is_sec == 1:
                                    nodes[parent].child_1 = local_root[s]
                                else:
                                    nodes[parent].child_0 = local_root[s]
                            continue
                        else:
                            sp = stack_ptr[None]
                            stack[sp][0] = mid
                            stack[sp][1] = e
                            stack[sp][2] = cur_node_idx
                            stack[sp][3] = 1
                            stack_ptr[None] += 1
                            sp = stack_ptr[None]
                            stack[sp][0] = s
                            stack[sp][1] = mid
                            stack[sp][2] = cur_node_idx
                            stack[sp][3] = 0
                            stack_ptr[None] += 1
                            nodes[cur_node_idx].init_interior(axis, -1, -1, temp_bounds)
                    # End of actual SAH partition branch.
                res = local_root_of_build
    return res


@ti.kernel
def build_hlbvh(
    primitives: ti.template(),           # original geometry
    bvh_primitives: ti.template(),
    ordered_prims: ti.template(),        # final re-ordered geometry array
    n_prims_in: ti.i32,
    nodes: ti.template(),
    total_nodes: ti.template(),
    ordered_prims_offset: ti.template(),
    max_prims_in_node: ti.template(),
    root_out: ti.template(),
):
    # 1) Compute centroid bounds.
    cb_min = vec3([INF, INF, INF])
    cb_max = vec3([-INF, -INF, -INF])
    for i in range(n_prims_in):
        c = bvh_primitives[i].bounds.centroid
        cb_min = ti.min(cb_min, c)
        cb_max = ti.max(cb_max, c)
    centroid_bounds_min[None] = cb_min
    centroid_bounds_max[None] = cb_max

    # 2) Compute Morton codes.
    n_morton[None] = n_prims_in
    scale = 1024.0
    dd = cb_max - cb_min
    for i in range(n_prims_in):
        c = bvh_primitives[i].bounds.centroid
        offset = c - cb_min
        x = offset[0] * scale / (dd[0] if dd[0] != 0 else 1e-8)
        y = offset[1] * scale / (dd[1] if dd[1] != 0 else 1e-8)
        z = offset[2] * scale / (dd[2] if dd[2] != 0 else 1e-8)
        code = encode_morton3(x, y, z)
        morton_prims[i].prim_idx = bvh_primitives[i].prim_num
        morton_prims[i].morton_code = code

    # 3) Radix sort.
    morton_radix_sort(n_morton[None])

    # 4) Group into LBVHTreelets by top bits.
    top_bits_mask = 0b00111111111111000000000000000000
    n_treelets[None] = 0
    start = 0
    for end in range(1, n_prims_in + 1):
        if end == n_prims_in:
            nt = n_treelets[None]
            treelets[nt].start_index = start
            treelets[nt].n_primitives = end - start
            treelets[nt].build_nodes_start = 0
            n_treelets[None] += 1
        else:
            c0 = morton_prims[start].morton_code & top_bits_mask
            c1 = morton_prims[end].morton_code & top_bits_mask
            if c0 != c1:
                nt = n_treelets[None]
                treelets[nt].start_index = start
                treelets[nt].n_primitives = end - start
                treelets[nt].build_nodes_start = 0
                n_treelets[None] += 1
                start = end

    # 5) Build LBVH for each treelet.
    total_nodes[None] = 0
    ordered_prims_offset[None] = 0
    for i in range(n_treelets[None]):
        st = treelets[i]
        bit_index = 17  # PBRT typical (29 - 12)
        root_idx = emit_lbvhtreelet(
            st.start_index,
            st.n_primitives,
            bit_index,
            bvh_primitives,
            primitives,
            ordered_prims,
            total_nodes,
            ordered_prims_offset,
            nodes,
            max_prims_in_node,
            emit_stack,
            emit_stack_ptr
        )
        treelet_roots[i] = root_idx

    # 6) Unify treelets with build_upper_sah.
    if n_treelets[None] == 1:
        root_out[None] = treelet_roots[0]
    else:
        root_val = build_upper_sah(
            nodes,
            total_nodes,
            n_treelets[None],
            max_prims_in_node,
            bucket_count_hlbvh,
            bucket_bmin_hlbvh,
            bucket_bmax_hlbvh,
            cost_arr_hlbvh,
            upper_sah_stack,
            upper_sah_stack_ptr,
            build_upper_sah_root,
            build_upper_sah_minp,
            build_upper_sah_maxp,
            build_upper_sah_centroid
        )
        root_out[None] = root_val
