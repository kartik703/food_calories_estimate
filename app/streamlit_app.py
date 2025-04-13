import streamlit as st
from PIL import Image
import os
import torch
from torchvision import transforms, models, datasets
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 101)
model = model.to(device)

# Load model weights
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "food_classifier.pth"))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load class labels
class_names = datasets.Food101(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")), download=False).classes

# Load calorie data
calorie_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "food_calories.csv")))

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred_index = output.argmax(1).item()
        food_name = class_names[pred_index]
    
    calories = calorie_df.loc[calorie_df['food_name'] == food_name, 'avg_calories'].values
    calories = int(calories[0]) if len(calories) > 0 else "N/A"

    return food_name, calories

# Streamlit UI
st.set_page_config(page_title="🍽️ Food Calorie Estimator", layout="centered")

st.title("🍱 AI Food Calorie Estimator")
st.caption("Upload a food image to detect the food type and estimate calories.")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        food, calories = predict(image)
    
    st.success(f"🍽️ Food: **{food}**")
    st.info(f"🔥 Estimated Calories: **{calories} kcal**")
