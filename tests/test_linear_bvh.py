import taichi as ti
from taichi.math import vec3

from accelerators.bvh import LinearBVHNode
from primitives.aabb import AABB  # your AABB dataclass

# Helper function to check if parent AABB completely contains child AABB.
@ti.func
def aabb_contains(parent: AABB, child: AABB) -> ti.i32:
    ret = 1
    for i in ti.static(range(3)):
        if parent.min_point[i] > child.min_point[i]:
            ret = 0
        elif parent.max_point[i] < child.max_point[i]:
            ret = 0
    return ret

@ti.kernel
def test_flattened_bvh(
    linear_bvh: ti.template(),  # your flattened nodes array
    bvh_stack: ti.template(),   # a stack for traversal
    stack_ptr: ti.template(),   # int32, shape=(1,)
    error_count: ti.template()  # int32, shape=(1,)
):
    """
    Checks the consistency of a flattened (LinearBVHNode) array by traversing
    from node 0. Example fields for each node might be:

        node.bounds     -> AABB
        node.axis       -> int (0..2)
        node.n_primitives -> int (>0 => leaf, 0 => interior)
        node.primitives_offset -> int (leaf data)
        node.second_child_offset -> int (for interior)
        # 'child_0' is assumed to be i+1 if interior and n_primitives=0
        # 'child_1' is second_child_offset.

    We'll do a stack-based traversal. For each interior node i:
        child_0 = i + 1
        child_1 = node.second_child_offset
    Then we check bounding box containment, etc.
    """

    error_count[0] = 0
    stack_ptr[0]   = 0

    # Start with root node index=0
    bvh_stack[0]   = 0
    stack_ptr[0]   = 1

    # Loop until no more nodes on stack
    while stack_ptr[0] > 0:
        stack_ptr[0] -= 1
        curr_idx = bvh_stack[stack_ptr[0]]

        node = linear_bvh[curr_idx]

        # If it's a leaf node => n_primitives > 0
        if node.n_primitives > 0:
            # LEAF checks:
            # e.g., make sure the offset is in valid range,
            # or if you want to verify partial things about leaf bounding boxes,
            # do that here. For now, we do minimal checks.
            pass

        else:
            # INTERIOR node => check children
            child_0 = curr_idx + 1
            child_1 = node.second_child_offset

            # Basic validity checks
            if child_0 < 0:
                error_count[0] += 1
            if child_1 < 0:
                error_count[0] += 1

            # Check axis
            if node.axis < 0 or node.axis > 2:
                error_count[0] += 1

            # Check bounding box containment
            if aabb_contains(node.bounds, linear_bvh[child_0].bounds) == 0:
                error_count[0] += 1
            if aabb_contains(node.bounds, linear_bvh[child_1].bounds) == 0:
                error_count[0] += 1

            # Push the children
            bvh_stack[stack_ptr[0]] = child_0
            stack_ptr[0] += 1
            bvh_stack[stack_ptr[0]] = child_1
            stack_ptr[0] += 1

    # Print error count for debugging
    print("Flattened BVH test error count =", error_count[0])




@ti.kernel
def print_flattened_bvh(flat_bvh: ti.template(), num_nodes: ti.i32):
    """
    Print every LinearBVHNode in the flattened BVH array.
    """
    for i in range(num_nodes):
        node = flat_bvh[i]
        if node.n_primitives > 0:
            print("LinearBVHNode", i, "[Leaf]: primitives_offset =", node.primitives_offset,
                     "n_primitives =", node.n_primitives,
                     "bounds = (min:", node.bounds.min_point, ", max:", node.bounds.max_point, ")")
        else:
            print("LinearBVHNode", i, "[Interior]: axis =", node.axis,
                     "second_child_offset =", node.second_child_offset,
                     "bounds = (min:", node.bounds.min_point, ", max:", node.bounds.max_point, ")")


