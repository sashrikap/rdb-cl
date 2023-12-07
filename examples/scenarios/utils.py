
def format_weights_dict(input_dict):
    print(f"input_dict: {input_dict}")
    weights = {}
    for key in ["dist_cars", "dist_lanes", "dist_fences", 
                "dist_obstacles", "dist_trees", "speed", "control"]:
        weights[key] = input_dict[key] if key in input_dict else 0.1
    return weights