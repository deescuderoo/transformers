import re, os
import numpy as np
from glob import glob

# Initialize a dictionary to hold lists of values for each layer and position
layer_position_values = {layer: [[] for _ in range(12)] for layer in range(12)}
OUTPUT_FOLDER = "output_cycles"

# Ensure the folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Find all files matching the pattern "output_max_cycle_*.txt" in the OUTPUT_FOLDER
files = glob(os.path.join(OUTPUT_FOLDER, "output_max_cycle_*.txt"))

# Regex pattern to extract layer number and values
pattern = re.compile(r"Layer (\d+): \[([-\d., ]+)\]")
count_corrrect=0
count_incorrrect=0
skipped_file = []
size=[]
# Process each file
for filename in files:
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                layer_id = int(match.group(1))
                values = list(map(float, match.group(2).split(',')))
                # Append each value to the corresponding position in `layer_position_values`
                if(len(values)==12):
                    count_corrrect=count_corrrect+1
                else:
                    count_incorrrect=count_incorrrect+1
                size.append(len(values))
                for pos, value in enumerate(values):
                    try:
                        layer_position_values[layer_id][pos].append(value)
                    except IndexError as e:
                        skipped_file.append(filename)

# Calculate min, max, and mean for each position in each layer across cycles
layer_stats = {layer: {'min': [], 'max': [], 'mean': []} for layer in range(12)}

for layer_id, positions in layer_position_values.items():
    for pos_values in positions:
        layer_stats[layer_id]['min'].append(float(np.min(pos_values)))
        layer_stats[layer_id]['max'].append(float(np.max(pos_values)))
        layer_stats[layer_id]['mean'].append(float(np.mean(pos_values)))

# Print results
for layer_id, stats in layer_stats.items():
    print(f"Layer {layer_id}:")
    print("Min values for each position:", stats['min'])
    print("Max values for each position:", stats['max'])
    print("Mean values for each position:", stats['mean'])
print("Correct files: ", count_corrrect)
print("Incorrect files: ", count_incorrrect)