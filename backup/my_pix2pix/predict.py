import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
from generator_model import Generator

# Load your trained model
checkpoint = torch.load("Models/v0.1/gen.pth.tar")
model = Generator(in_channels=3)
model.double()
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set the model to evaluation mode

# Load an input image in the (channels, length, breadth) format
input_image_path = "Task1/val/1BA075/mr.nii.gz"
nii_img = nib.load(input_image_path)
nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
input_image = np.array(nii_img.get_fdata())
mean = np.mean(input_image)
std = np.std(input_image)
norm_input = (input_image - mean) / std
input_image = input_image[:, : , 109:112]
input_image = input_image.astype(np.float64) / 255.0
# input_image = np.transpose(input_image, (2, 0, 1))
transform = transforms.ToTensor()
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

# Forward pass through the model
with torch.no_grad():
    output_tensor = model(input_tensor)

# Convert the output tensor to a NumPy array
output_array = output_tensor.squeeze().cpu().numpy()

# Transpose the output array back to (length, breadth, channels) format
output_array = np.transpose(output_array, (1, 2, 0))
# output_array = output_array * 1000.0

# Convert the output array to a PIL image
output_image = Image.fromarray((output_array).astype(np.float64))

# Save the output image as a .png file
output_image_path = "evaluation/output_image_test.png"
output_image.save(output_image_path)
