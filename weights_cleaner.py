import torch

checkpoint = torch.load('./weights/best.pt', map_location='cpu')

# Print the type to understand what's inside
print("Checkpoint keys:", checkpoint.keys())
print("Type of 'model':", type(checkpoint['model']))

torch.save({'model': checkpoint['model'].state_dict()}, 'best_clean.pt')