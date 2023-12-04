import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Grayscale
from datasets import load_dataset
from tqdm import tqdm

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, labels = batch['image'], batch['label']
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Create a function to apply the transforms and return images and labels
def transform_batch(batch):
    # Apply the transformations to the image in the batch
    transformed_image = data_transforms(batch['image'])
    label = batch['label']
    # Return a dictionary with the transformed image and its corresponding label
    return {'image': transformed_image, 'label': label}

if __name__ == '__main__':
    # set the random seeds for reproducibility
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Load the datasets
    test_dataset = load_dataset('imagenet-1k', split='test', streaming=True, use_auth_token=True)
    print('---------------------------')
    print(test_dataset)
    print(f'Number of test samples: 100,000')

    # Load the model
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc')
    dinov2_vitg14.linear_head = torch.nn.Linear(dinov2_vitg14.linear_head.in_features, 1000)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vitg14.to(device)

    print('---------------------------')
    print(dinov2_vitg14)

    # define the data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        Grayscale(num_output_channels=3),  # Convert grayscale images to pseudo-RGB
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create DataLoaders
    test_loader = DataLoader(test_dataset.map(transform_batch), batch_size=64)

    # Evaluate the model
    accuracy = evaluate(dinov2_vitg14, test_loader, device)

    # Print the output
    print('---------------------------')
    print(f'Test Accuracy of Dino V2: {accuracy}%')
    print('---------------------------')
