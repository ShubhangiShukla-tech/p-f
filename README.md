# Wafer Pass/Fail Detection App

A Streamlit-based app to detect whether a semiconductor wafer passes or fails quality inspection, based on uploaded wafer map images using a ResNet18 model.

## 🧪 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/wafer-pass-fail-app.git
cd wafer-pass-fail-app
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your trained model file `best_model.pth` to the project root.

4. Start the Streamlit app
```bash
streamlit run app.py
```

## ✅ Features

- Upload and display wafer map images.
- Predict pass/fail using a trained deep learning model.
- Interactive and aesthetic UI built with Streamlit.