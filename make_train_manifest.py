import json
import os

# Input file paths
train_json_path = '/mnt/e/genvfi_outputs/results/amt/lavib_subset_7/results.json'
test_json_path = '/mnt/e/genvfi_outputs/results/amt/davis90/results.json'

# Output structure
output_data = {"train": {}, "test": {}}

# Load and process train data
with open(train_json_path, 'r') as f:
    train_data = json.load(f)

for entry in train_data:
    lpips = entry.get('lpips', 0)
    flolpips = entry.get('flolpips', 0)
    if lpips > 0.1 or flolpips > 0.1:
        data_id = entry['video']
        output_data['train'][data_id] = {
            "image": entry['pred_path'],
            "target_image": entry['gt_path'],
            "ref_image": entry['frame0_path'],
            "prompt": "remove degradation"
        }

# Load and process test data
with open(test_json_path, 'r') as f:
    test_data = json.load(f)

for entry in test_data:
    lpips = entry.get('lpips', 0)
    flolpips = entry.get('flolpips', 0)
    if lpips > 0.1 or flolpips > 0.1:
        data_id = entry['video']
        prediction_path = entry['prediction'].replace('/home/cx24957/VFI/GenVFI/', '/mnt/e/genvfi_outputs/')
        output_data['test'][data_id] = {
            "image": prediction_path,
            "target_image": entry['gt'],
            "ref_image": entry['frame0'],
            "prompt": "remove degradation"
        }

# Save to output JSON
output_path = '/mnt/e/Difix3d/data/converted_dataset.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Converted JSON saved to {output_path}")
