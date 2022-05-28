dataset = BrainTumorDataset(dataset_path=dataset_path)

mean = std = 0

sum = 0

for x, y, _ in dataset:
    sum += x
