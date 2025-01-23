import re, os
import numpy as np
from glob import glob

# # Initialize a dictionary to hold lists of values for each layer and position
# layer_position_values = {layer: [[] for _ in range(12)] for layer in range(12)}
# OUTPUT_FOLDER = "output_cycles_2"
#
# # Ensure the folder exists
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#
# # Find all files matching the pattern "output_max_cycle_*.txt" in the OUTPUT_FOLDER
# files = glob(os.path.join(OUTPUT_FOLDER, "output_max_cycle_*.txt"))
#
# # Regex pattern to extract layer number and values
# pattern = re.compile(r"Layer (\d+): \[([-\d., ]+)\]")
# skipped_file = []
# size=[]
# # Process each file
# for filename in files:
#     with open(filename, 'r') as file:
#         for line in file:
#             match = pattern.search(line)
#             if match:
#                 layer_id = int(match.group(1))
#                 values = list(map(float, match.group(2).split(',')))
#                 # Append each value to the corresponding position in `layer_position_values`
#                 for pos, value in enumerate(values):
#                     try:
#                         layer_position_values[layer_id][pos].append(value)
#                     except IndexError as e:
#                         skipped_file.append(filename)
#
# # Calculate min, max, and mean for each position in each layer across cycles
# layer_stats = {layer: {'min': [], 'max': [], 'mean': []} for layer in range(12)}
#
# for layer_id, positions in layer_position_values.items():
#     for pos_values in positions:
#         layer_stats[layer_id]['min'].append(float(np.min(pos_values)))
#         layer_stats[layer_id]['max'].append(float(np.max(pos_values)))
#         layer_stats[layer_id]['mean'].append(float(np.mean(pos_values)))
#
# # Print results
# for layer_id, stats in layer_stats.items():
#     print(f"Layer {layer_id}:")
#     print("Min values for each position:", stats['min'])
#     print("Max values for each position:", stats['max'])
#     print("Mean values for each position:", stats['mean'])

def compute_fallback_from_file(file_path):
    """
    Computes fallback mean, max, and min values from the shape mismatch log file.

    Args:
        file_path (str): Path to the shape mismatch log file.

    Returns:
        dict: A dictionary containing the fallback mean, max, and min values.
    """
    means = []
    mins = []
    maxs = []

    # Regex patterns to extract summary statistics
    summary_pattern = re.compile(r"mean=([-\d.]+), min=([-\d.]+), max=([-\d.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = summary_pattern.search(line)
            if match:
                # Extract mean, min, and max from the line
                means.append(float(match.group(1)))
                mins.append(float(match.group(2)))
                maxs.append(float(match.group(3)))

    # Compute overall fallback values
    fallback_mean = np.mean(means) if means else 0.0
    fallback_max = np.max(maxs) if maxs else 0.0
    fallback_min = np.min(mins) if mins else 0.0

    return {
        "fallback_mean": fallback_mean,
        "fallback_max": fallback_max,
        "fallback_min": fallback_min
    }


# Example usage
if __name__ == "__main__":
    file_path = "output_cycles/shape_mismatch_log.txt"  # Replace with the actual file path
    fallback_values = compute_fallback_from_file(file_path)

    print("Fallback values:")
    print(f"Mean: {fallback_values['fallback_mean']}")
    print(f"Max: {fallback_values['fallback_max']}")
    print(f"Min: {fallback_values['fallback_min']}")