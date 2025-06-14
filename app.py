import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

st.set_page_config(page_title="Wafer Pass/Fail Detection", layout="centered")
st.title("🔍 Wafer Pass/Fail Detector")
st.markdown("Upload a wafer image and find out if it's **✅ PASS** or **❌ FAIL**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("Analyzing..."):
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    result = "✅ PASS" if prediction == 0 else "❌ FAIL"
    st.markdown(f"### Result: {result}")