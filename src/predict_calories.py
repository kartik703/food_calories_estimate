import torch
from torchvision import models, transforms, datasets
from PIL import Image
import pandas as pd
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 101)
model = model.to(device)

# 🔒 Load model weights using absolute path
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "food_classifier.pth"))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load Food101 class names
class_names = datasets.Food101(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")), download=False).classes

# Load calorie mapping
calorie_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "food_calories.csv"))
calorie_df = pd.read_csv(calorie_path)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_index = outputs.argmax(1).item()
        predicted_class = class_names[predicted_index]

    # Look up calories
    calories = calorie_df.loc[calorie_df['food_name'] == predicted_class, 'avg_calories'].values
    calories = int(calories[0]) if len(calories) > 0 else "N/A"

    return predicted_class, calories
