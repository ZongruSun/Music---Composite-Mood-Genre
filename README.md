
Human is always self-conflicted, so the art work.....

## 🎵 Project Overview
This project explores **Composite Mood Genre (CMG) classification in music** using **Convolutional Neural Networks (CNNs)**. Traditional music classification often assigns a **single mood tag** (e.g., happy or sad), which fails to capture the complexity of human emotions. Our model aims to classify songs into **Composite Mood Genres**, such as "Relax but Energetic" or "Negative but Hopeful" to improve music recommendations.

## 🚀 Key Features
- **Deep Learning Model**: Uses **CNN-64** and **LeNet-5** for classifying music moods.
- **Spectrogram-Based Approach**: Converts audio files into spectrogram images for CNN processing.
- **Custom Mood Labels**: Songs are categorized into **10 composite mood genres**.
- **Music Recommendation Potential**: Enhances music streaming platforms with **emotionally intelligent recommendations**.

## 📂 Key Results
<img width="385" alt="image" src="https://github.com/user-attachments/assets/32379f83-e63c-40ac-bed8-1a719fdabf47" /><img width="385" alt="image" src="https://github.com/user-attachments/assets/39f58116-be6b-44d2-8f97-16d7610b8fd9" />



---

## 📂 File Structure
```plaintext
📦 Project Structure
│── Composite Mood Classification.ipynb  # Jupyter notebook for training & testing models
│── Data/
│   ├── CalmButEpic/
│   ├── CalmButWarm/
│   ├── ConcernedButLight/
│   ├── InsecureButInessential/
│   ├── NegativeButHopeful/
│   ├── Other/
│   ├── PositiveButVoid/
│   ├── RelaxButEnergetic/
│   ├── TotallyNegative/
│   ├── TotallyPositive/
│── Final Report.pdf  # Research paper summarizing methodology & results
│── requirements.txt  # Dependencies for the project
```

---

## 🛠️ Installation & Setup
### **1️⃣ Clone the Repository & Navigate to Directory**
```bash
git clone https://github.com/ZongruSun/Composite-Mood-Classification.git
cd Composite-Mood-Classification
```

### **2️⃣ Install Python Dependencies**
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

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

---

## 📸 **Application Screenshot**
Here is a screenshot of the working **Composite Mood Classification System**:

![App Screenshot](./path-to-your-screenshot.png)

### **🔹 How It Works**
1. **Upload an audio file**.
2. **The system converts it to a spectrogram**.
3. **CNN predicts the composite mood genre**.
4. **Results are displayed with confidence scores**.

---



