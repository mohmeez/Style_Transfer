# Neural Style Transfer App

This project implements a Neural Style Transfer model using PyTorch and Streamlit. The app allows users to upload a content image and a style image, then combines them to generate a new image where the style of the second image is transferred to the first one.
Model Description

The style transfer model is built using the VGG19 pre-trained convolutional neural network from PyTorch's torchvision library. The VGG19 network is used as the backbone for extracting features from the content and style images. The style transfer process involves minimizing the difference between the content and style representations of the images using content loss and style loss functions.

### Key components:

  **Content Loss:** Measures the difference between the high-level feature maps of the content image and the generated image.
  
  **Style Loss:** Compares the correlation of feature maps (using the Gram matrix) of the style image and the generated image.
  
  The model uses LBFGS optimizer for efficient convergence.
      
## How to Run the App Locally

  CLone the Repo:
      
      git clone https://github.com/mohmeez/Style_Transfer.git
       cd Style_Transfer
## Set up the environment:

    Install Python 3.x if not already installed.
    Create a virtual environment and activate it:
    
## Install the required dependencies:
    pip install -r requirements.txt
## Run the Streamlit app:
    streamlit run app.py
### Upload your content and style images through the app interface, and the neural style transfer will generate a new image

## Deployed Streamlit App
    https://styletransfer-9sd5lqk76pzqdodjtnrpci.streamlit.app/
## Sample Image:
  ![Screenshot from 2024-10-17 15-28-54](https://github.com/user-attachments/assets/9f752568-5fdc-4350-b41a-11601ea13d86)
