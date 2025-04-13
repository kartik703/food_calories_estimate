🍱 Food Calorie Estimator (AI-Powered)

This project uses computer vision to detect food items in images and estimate their average calories using a deep learning model trained on the Food101 dataset. The app is built with PyTorch, EfficientNet, and Streamlit — and supports GPU acceleration.
---

## 🚀 Features

- 🔍 **Food Recognition** using EfficientNet-B0 (transfer learning)
- ⚡ **Fast GPU inference** (CUDA-enabled)
- 📷 **Upload image**, get calorie predictions in real-time
- 🍔 Based on **Food101 dataset** and a custom calorie mapping
- 💻 Deployable via **Streamlit**, works locally or on Hugging Face

---

## 📁 Project Structure

```
food_calorie_estimator/
├── app/                 # Streamlit app
├── data/                # Food101 dataset and food_calories.csv
├── models/              # Trained PyTorch model (food_classifier.pth)
├── notebooks/           # Jupyter notebooks for training & EDA
├── src/                 # Model + prediction logic
├── venv/                # Virtual environment (not tracked in Git)
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/kartik703/food_calories_estimate.git
cd food_calorie_estimator
```

### 2. Create virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training (Optional)

The model is trained on Food101 using EfficientNet-B0:

```python
from torchvision import models
model = models.efficientnet_b0(pretrained=True)
```

Trained weights are saved in:

```
models/food_classifier.pth
```

You can retrain using `notebooks/02_food_classifier_training.ipynb`.

---

## 🖼️ Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Sample Prediction

Upload a food image like this:

- 🍕 `pizza.jpg` → **410 kcal**
- 🥗 `caesar_salad.jpg` → **270 kcal**
- 🍩 `donuts.jpg` → **350 kcal**

---

## 📊 Tech Stack

- Python
- PyTorch
- torchvision
- Streamlit
- Food101 Dataset

---

## 💡 Future Ideas

- Add nutrition breakdown (protein/fat/carbs)
- Detect multiple foods per image (object detection)
- Deploy to Hugging Face Spaces
- Add calorie tracking history (with charts)

---

## 🙌 Credits

- [Food101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- PyTorch / torchvision
- Streamlit Team

---

## 🧑‍💻 Author

**Kartik Goswami**  
[MSc Data Science & AI | Newcastle University]

Feel free to ⭐ this repo and share if you found it useful!
```