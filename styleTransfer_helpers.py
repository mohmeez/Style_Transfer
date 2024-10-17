from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch

# Determine if GPU (CUDA) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the image, resize it, and apply transformations for normalization
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')  # Load the image and ensure it's in RGB mode
    size = min(max_size, max(image.size))  # Resize based on the smaller dimension while maintaining aspect ratio
    # Define the necessary transformations (resize, convert to tensor, normalize using ImageNet stats)
    transformations = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

    # Apply transformations and add a batch dimension for compatibility with the model
    image = transformations(image).unsqueeze(0)
    return image.to(device)

# Custom content loss class, comparing input features with target content features
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Store the target content feature map (detached from the graph)
        self.target = target.detach()

    def forward(self, input):
        # Compute mean squared error between input and target content
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Custom style loss class, comparing the input and target using the Gram matrix
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Compute the target's Gram matrix
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        # Calculate Gram matrix (correlations between features)
        a, b, c, d = input.size()  # batch, channels, height, width
        features = input.view(a * b, c * d)  # Flatten the spatial dimensions
        G = torch.mm(features, features.t())  # Compute Gram matrix
        return G.div(a * b * c * d)  # Normalize by the number of elements

    def forward(self, input):
        # Compute the Gram matrix of the input and the style loss
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Function to create a model with style and content loss layers inserted
def get_style_model_and_losses(cnn, content_image, style_image):
    content_layers = ['conv_4']  # Define which layers are used for content loss
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # Define layers used for style loss

    model = nn.Sequential()  # Build the model incrementally
    content_losses = []
    style_losses = []

    i = 0
    # Iterate through the layers of the pretrained CNN (VGG19)
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)  # Replace with non-in-place ReLU to avoid modification of input
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)  # Add the layer to the model

        if name in content_layers:
            target = model(content_image).detach()  # Pass content image through the model up to this layer
            content_loss = ContentLoss(target)  # Create and append content loss
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()  # Pass style image through the model up to this layer
            style_loss = StyleLoss(target_feature)  # Create and append style loss
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    return model, content_losses, style_losses  # Return the model with the loss layers

# Style transfer process that iteratively updates the input image to match content/style targets
def run_style_transfer(model, content_losses, style_losses, input_image, num_steps=300):
    optimizer = optim.LBFGS([input_image.requires_grad_()])  # Use L-BFGS optimizer

    run = [0]
    while run[0] <= num_steps:
        # Closure function for optimization step
        def closure():
            input_image.data.clamp_(0, 1)  # Ensure pixel values remain in valid range
            optimizer.zero_grad()  # Clear gradients
            model(input_image)  # Forward pass through the model

            # Compute total content and style losses
            content_score = sum([cl.loss for cl in content_losses])
            style_score = sum([sl.loss for sl in style_losses])

            # Adjust style weight and compute total loss
            total_loss = content_score + (style_score * 1e2)
            total_loss.backward()  # Backpropagate the loss

            run[0] += 1  # Increment step count
            return total_loss

        optimizer.step(closure)  # Perform optimization step

    input_image.data.clamp_(0, 1)  # Clamp final image values
    return input_image  # Return the final styled image
