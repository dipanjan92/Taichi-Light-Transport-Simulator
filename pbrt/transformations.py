def parse_transform(transform_list):
    if not transform_list or not isinstance(transform_list, list):
        raise ValueError("Expected a non-empty list for transform")

    # Get the first (and assumed only) transform dictionary.
    transform_entry = transform_list[0]
    properties = transform_entry.get("properties", {})
    matrix = properties.get("matrix")
    if not matrix or len(matrix) != 16:
        raise ValueError("Transform matrix must be a list of 16 numbers")

    # Convert to tuple (or you can use your set_matrix() function here)
    return tuple(float(x) for x in matrix)