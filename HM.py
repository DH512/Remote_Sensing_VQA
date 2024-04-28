import json
file_path = 'modified_grouped_by_img_id_random_match.json'
# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)
total_zeros = 0
total_img_ids = 0

for entries in data.values():
    total_img_ids += 1  # Count each group of entries under each img_id as one img_id
    total_zeros += sum(1 for entry in entries if entry['match'] == 0)

# Calculate the ratio of total zeros to the number of img_id
ratio = total_zeros / total_img_ids if total_img_ids > 0 else 0


print(f"The Hamming Distance IS: {ratio:.2f}")