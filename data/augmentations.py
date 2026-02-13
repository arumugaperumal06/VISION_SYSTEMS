from torchvision import transforms
import torchvision.transforms.v2 as v2

def get_transforms():
   
    clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
   
    drift = transforms.Compose([
        transforms.Resize((224, 224)),
        v2.ColorJitter(brightness=0.4, contrast=0.3),
        v2.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
        transforms.ToTensor(),
        v2.GaussianNoise(mean=0, sigma=0.07),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return clean, drift