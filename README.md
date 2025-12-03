# Painting Era Classifier 𐔌՞. .՞𐦯
Identifying artistic periods using a deep learning model.

# Overview
This web app uses a TensorFlow image classifier deployed in Streamlit to identify whether a painting belongs to Baroque, Medieval, or Renaissance era.
Users can upload a painting image and instantly see:
- Predicted painting era
- Confidence score
- Short description
- Gold-themed UI
- Confidence bar chart (Altair)

# Features
- EfficientNet-based image classifier
- Custom styled UI
- Live prediction and confidence chart
- Supports JPG / JPEG / PNG

# Model
Raw Dataset: Painting Eras Detection Classification Dataset by ArtAncestry
https://share.google/rqdLnLG0PWmTDK0zf

Trained Dataset:
https://drive.google.com/drive/folders/1heZ_RkMHTFwX5y_xuMbqNWr0fuvIr1w8?usp=sharing

Trained on 3 classes:
- Baroque paintings
- Medieval art
- Renaissance paintings

Preprocessing:
- Resize to 224×224
- EfficientNet preprocessing
- Saved in TensorFlow SavedModel format

# Installation
1️⃣ Clone the repository:
- git clone https://github.com/babytokki/painting-era-classifier.git
- cd painting-era-classifier

2️⃣ Install dependencies:
- pip install -r requirements.txt

▶️ Run the App:
- run main.ipynb
- streamlit run app.py
