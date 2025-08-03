🤖 Hand Gesture Recognition using CNN

This project is part of my **Prodigy Infotech Internship**, where I developed a deep learning-based system that recognizes hand gestures in real time using a **Convolutional Neural Network (CNN)**.

---
🎯 Project Objective

To create a gesture recognition model that can classify hand signs from image data using a CNN trained on labeled gesture datasets. It enables applications in Human-Computer Interaction (HCI), sign language recognition, and more.

---

 🧠 Key Features

✅ Built a CNN from scratch for gesture classification  
✅ Trained the model on hand gesture image dataset  
✅ Achieved high accuracy on both training & validation sets  
✅ Real-time gesture detection using OpenCV & camera feed  
✅ Visualized prediction results using Matplotlib

---

 📂 Dataset Used

- Hand Gesture dataset (can be custom or Kaggle-based)
- Classes include: ✋ Palm, 👊 Fist, 🤙 Call Me, 👌 OK, ✌️ Victory, etc.
- Preprocessed: Grayscale conversion, resizing (e.g., 64x64), normalization

---

 🔧 Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Matplotlib  

---

 📊 Model Architecture

- Input Layer (64x64x1)  
- Conv2D → ReLU → MaxPooling  
- Conv2D → ReLU → MaxPooling  
- Flatten → Dense → Softmax

---

 📁 Repository Structure

```bash
.
├── gesture_model.ipynb         # Jupyter Notebook with full pipeline
├── gestures_dataset/           # Training images
├── saved_model/                # Trained CNN model
├── real_time_prediction.py     # Real-time detection via webcam
├── README.md                   # Project Documentation
