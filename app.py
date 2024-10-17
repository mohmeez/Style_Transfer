import streamlit as st
from styleTransfer_helpers import load_image, get_style_model_and_losses, run_style_transfer, vgg19, VGG19_Weights
from PIL import Image
from torchvision import transforms
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Title for the Streamlit app
st.title("Neural Style Transfer")


# File uploader for content and style images
content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# Once both images are uploaded, proceed with the style transfer
if content_image_file and style_image_file:
    content_image_path = f"temp_content_image.jpg"
    style_image_path = f"temp_style_image.jpg"

    # Save the uploaded files locally
    with open(content_image_path, "wb") as f:
        f.write(content_image_file.getbuffer())
    with open(style_image_path, "wb") as f:
        f.write(style_image_file.getbuffer())

    # Load VGG19 model and move it to GPU (or CPU if unavailable)
    cnn = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    # Load the content and style images and move them to GPU (or CPU if unavailable)
    original_image = Image.open(content_image_path)
    original_size = original_image.size  # Store the original size as (width, height)

    content_image = load_image(content_image_path)  # Content image moved to GPU
    style_image = load_image(style_image_path)  # Style image moved to GPU

    # Display the content and style images side by side 
    st.image([Image.open(content_image_path), Image.open(style_image_path)], 
             caption=["Content Image", "Style Image"], width=300)

    # Perform style transfer by building the model with the loss layers 
    model, content_losses, style_losses = get_style_model_and_losses(cnn, content_image, style_image)
    output_image = run_style_transfer(model, content_losses, style_losses, content_image.clone().to(device))

    # Convert the tensor back to an image
    output_image = output_image.cpu().clone().squeeze(0)  # Move output image back to CPU

    # Handle any potential NaN or Inf values ( since we had a few NAN values)
    output_image[torch.isnan(output_image)] = 0
    output_image[torch.isinf(output_image)] = 0

    # Clamp values to the range [0, 1] to ensure valid pixel values
    output_image = torch.clamp(output_image, 0, 1)

    # Convert the tensor to a PIL image for display
    unloader = transforms.ToPILImage()
    output_image = unloader(output_image)

    # Resize the output image to the original content image size
    output_image = output_image.resize(original_size, Image.LANCZOS)

    # Display the final styled image
    st.image(output_image, caption="Styled Image", width=300)
