from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the image and apply transformations
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = min(max_size, max(image.size))
    transformations = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    image = transformations(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


# Content loss calculation class
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


# Style loss calculation class
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


# Function to get the model with content and style losses
def get_style_model_and_losses(cnn, content_image, style_image):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    return model, content_losses, style_losses


# Style transfer function
def run_style_transfer(model, content_losses, style_losses, input_image, num_steps=300):
    optimizer = optim.LBFGS([input_image.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_image.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_image)

            content_score = sum([cl.loss for cl in content_losses])
            style_score = sum([sl.loss for sl in style_losses])

            total_loss = content_score + (style_score * 1e1)  # Adjust style weight
            total_loss.backward()

            run[0] += 1
            return total_loss

        optimizer.step(closure)

    input_image.data.clamp_(0, 1)
    return input_image


