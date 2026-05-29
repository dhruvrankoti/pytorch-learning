import streamlit as st
import torch
from torch import nn
from torchvision.transforms import v2
from PIL import Image

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

transform = v2.Compose([
    v2.Resize((28, 28)),
    v2.Grayscale(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

st.title("FashionMNIST Classifier")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
    
    st.success(f"Prediction: {classes[pred]}")
    st.subheader("Probabilities")
    for i, label in enumerate(classes):
        st.write(f"{label}: {probs[0, i].item():>.4f}")