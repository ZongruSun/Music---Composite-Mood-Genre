
Human is always self-conflicted, so the artwork.....\
Let's start with solo I created below, can you tell me what's your feeling?



Yes, so the single mood genres, which are used by Spotify and Apple Music, cannot capture our true feelings - the most import for Art\
Now, here comes this project


https://github.com/user-attachments/assets/2783b829-4c8c-4d38-b709-3a3c8fb4557d




## 🎵 Project Overview
This project explores **Composite Mood Genre (CMG) classification in music** using **Convolutional Neural Networks (CNNs)**. Traditional music classification often assigns a **single mood tag** (e.g., happy or sad), which fails to capture the complexity of human emotions. Our model aims to classify songs into **Composite Mood Genres**, such as "Relax but Energetic" or "Negative but Hopeful" to improve music recommendations.

## 🚀 Key Features
- **Deep Learning Model**: Uses **CNN-64** and **LeNet-5** for classifying music moods.
- **Spectrogram-Based Approach**: Converts audio files into spectrogram images for CNN processing.
- **Custom Mood Labels**: Songs are categorized into **10 composite mood genres**.
- **Music Recommendation Potential**: Enhances music streaming platforms with **emotionally intelligent recommendations**.

## 📂 Key Results
<img width="385" alt="image" src="https://github.com/user-attachments/assets/32379f83-e63c-40ac-bed8-1a719fdabf47" /><img width="394" alt="image" src="https://github.com/user-attachments/assets/39f58116-be6b-44d2-8f97-16d7610b8fd9" />
### **🔹 Insights from the Results**
- **CNN-64 outperforms LeNet-5**: The **CNN-64 model** shows **higher classification accuracy** across most composite mood genres compared to LeNet-5, as seen in the heatmap.
- **Stronger classification for dominant moods**: Categories like **"Relax but Energetic"** and **"Insecure but Inessential"** show higher correct predictions, indicating these moods have distinct features that are well captured by the model.
- **Higher recall in CNN-64**: The **right-side heatmap** (CNN-64) shows higher numbers along the diagonal, suggesting better **recall performance** compared to LeNet-5.
- **Confusion in closely related moods**: Some categories, like **"Calm but Warm"** and **"Calm but Epic"**, show moderate misclassification, likely due to overlapping musical features.
- **Improved generalization**: The **"Other"** category has better distribution in CNN-64, indicating it is **less likely to classify ambiguous inputs into dominant classes**, improving generalization.

---

## 📌 **Usage Guide**

### **1️⃣ Convert Audio Files to Spectrograms**
Run the following script to preprocess the audio data:
```python
from utils import generate_spectrogram

generate_spectrogram("path/to/audio/file.wav", "output/spectrogram.png")
```

### **2️⃣ Train the CNN Model**
```python
python train_cnn.py
```
- Loads dataset & generates spectrogram images.
- Trains **CNN-64 & LeNet-5** for music mood classification.
- Saves the trained model for inference.

### **3️⃣ Evaluate Model Performance**
```python
python evaluate_model.py
```
- Computes **Accuracy & Recall metrics**.
- Generates **Confusion Matrix & Heatmap**.

---

## 📊 **Model Architecture**
The project uses **CNN-based architectures** for feature extraction from spectrograms.

### **🔹 CNN-64 Model**
- **Input**: Spectrogram images.
- **Convolution Layers**: Extract frequency-time domain features.
- **Fully Connected Layers**: Classifies the mood genre.
- **Output**: One of 10 composite mood genres.

```python
class CNN64(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN64, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

---

## 📌 **API & Code Example**
### **1️⃣ Calling the Model for Prediction**
```python
import torch
from model import CNN64

# Load Trained Model
model = CNN64()
model.load_state_dict(torch.load("cnn64_model.pth"))
model.eval()

# Perform Prediction
def predict_mood(spectrogram_path):
    image = preprocess_spectrogram(spectrogram_path)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

result = predict_mood("sample_spectrogram.png")
print(f"Predicted Mood: {result}")
```

---

## 📜 License
This project is open-sourced under the **MIT License**, and contributions are welcome!

## 🤝 Contributing
We welcome **Issues** and **Pull Requests**!
1. Fork this repository
2. Create a new branch (`git checkout -b feature-new-feature`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push the branch (`git push origin feature-new-feature`)
5. Submit a Pull Request

---

## 📧 Contact
For any questions or suggestions, feel free to contact [ZongruSun](https://github.com/ZongruSun) 🚀




