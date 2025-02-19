import taichi as ti
from taichi.math import vec3

from accelerators.bvh import BVHNode
from primitives.aabb import AABB  # your AABB dataclass

# Helper function: tests if a parent's AABB completely contains a child's AABB.
@ti.func
def aabb_contains(parent: AABB, child: AABB) -> ti.i32:
    ret = 1  # Assume containment is true
    for i in ti.static(range(3)):
        if parent.min_point[i] > child.min_point[i]:
            ret = 0
        elif parent.max_point[i] < child.max_point[i]:
            ret = 0
    return ret

# Taichi kernel to test the BVH (non-flattened) correctness.
# NOTE:
#   - bvh_nodes is assumed to be a 1D Taichi field of BVHNode.
#   - bvh_stack is a 1D int32 Taichi field used as the traversal stack.
#   - stack_ptr is an int32 Taichi field of shape (1,) holding the current stack pointer.
#   - error_count is an int32 Taichi field of shape (1,) that will count errors.
@ti.kernel
def test_bvh(bvh_nodes: ti.template(),
             bvh_stack: ti.template(),
             stack_ptr: ti.template(),
             error_count: ti.template()):
    # Initialize error count and the stack pointer.
    error_count[0] = 0
    stack_ptr[0] = 0

    # Push the root node index (assumed to be 0) onto the stack.
    bvh_stack[0] = 0
    stack_ptr[0] = 1

    # Process nodes until the stack is empty.
    while stack_ptr[0] > 0:
        # Pop the top of the stack.
        stack_ptr[0] -= 1
        curr_idx = bvh_stack[stack_ptr[0]]
        node = bvh_nodes[curr_idx]

        # Check if this is a leaf node.
        if node.n_primitives > 0:
            # Leaf nodes: you may add additional tests here, such as
            # checking that the primitive offset is valid.
            # (No child checks are needed for leaf nodes.)
            pass
        else:
            # Interior node: check that children indices are valid.
            # (In your BVH construction, children indices should have been set.)
            if node.child_0 < 0:
                error_count[0] += 1
            if node.child_1 < 0:
                error_count[0] += 1

            # Check that the split axis is in a valid range.
            if node.split_axis < 0 or node.split_axis > 2:
                error_count[0] += 1

            # Check that each child's AABB is contained within the parent's AABB.
            if aabb_contains(node.bounds, bvh_nodes[node.child_0].bounds) == 0:
                error_count[0] += 1
            if aabb_contains(node.bounds, bvh_nodes[node.child_1].bounds) == 0:
                error_count[0] += 1

            # Push the children indices onto the stack for further traversal.
            bvh_stack[stack_ptr[0]] = node.child_0
            stack_ptr[0] += 1
            bvh_stack[stack_ptr[0]] = node.child_1
            stack_ptr[0] += 1

    # Optionally, print the error count.
    print("BVH test error count =", error_count[0])
    






@ti.kernel
def print_bvh(bvh: ti.template(), num_nodes: ti.i32):
    """
    Print every BVHNode in the array with its key debug information.
    (This does not print the tree structure; it simply iterates over the nodes.)
    """
    for i in range(num_nodes):
        node = bvh[i]
        if node.n_primitives > 0:
            print("BVHNode", i, "[Leaf]: first_prim_offset =", node.first_prim_offset,
                     "n_primitives =", node.n_primitives,
                     "bounds = (min:", node.bounds.min_point, ", max:", node.bounds.max_point, ")")
        else:
            print("BVHNode", i, "[Interior]: split_axis =", node.split_axis,
                     "child_0 =", node.child_0, "child_1 =", node.child_1,
                     "bounds = (min:", node.bounds.min_point, ", max:", node.bounds.max_point, ")")

