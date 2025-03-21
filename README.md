# MobileNetV2 Android Deployment

## **1. Project Objective**
This project aims to deploy a pre-trained **MobileNetV2** deep learning model for image classification on an **Android** device. The model is converted from **PyTorch** to **TFLite** and integrated into an Android application, enabling users to classify images using their mobile phones.

## **2. Technical Structure**
### **2.1 Software Stack**
- **Model Training & Conversion**: PyTorch → ONNX → TensorFlow → TFLite
- **Mobile App Development**: Android Studio (Kotlin-based)
- **Model Deployment**: TensorFlow Lite (TFLite) on Android
- **Version Control**: Git, GitHub

![Model Design-2025-03-07-050838](https://github.com/user-attachments/assets/3074899e-40bc-4c2f-a894-0cd082b131f5)

### **2.2 Hardware**
- **Device**: Android smartphone (Samsung tested, API Level 31+)

### **2.3 Programming Languages**
- **Python** (Model training & conversion)
- **Kotlin** (Android application development)

## **3. How to Use the Files**

### **3.1 Python Files (Model Training & Conversion)**
- `MobileNetV2.py`: Loads and tests MobileNetV2 in PyTorch.
- `torch_to_tflite.py`: Converts PyTorch model to TFLite. This is the only file you need to run after installed requirements files.
- `imagenet_classes.txt`: Class labels for classification.
- `tensorflow_requirements.txt`: Dependencies for TensorFlow.
- `pytorch_requirements.txt`: Dependencies for PyTorch.
- `typing_extensions_requirements.txt`: Dependencies for typing_extensions.

### **3.2 Android Files (App Implementation)**
Create an new project in Android studio and replace or create below files. 
- `app/src/main/java/com/example/mobilenetv2/MainActivity.kt`: Main application logic.
- `app/src/main/res/layout/activity_main.xml`: UI layout.
- `app/src/main/AndroidManifest.xml`: App permissions and configuration.
- `app/build.gradle.kts`: Dependencies for TensorFlow Lite.
- `assets/mobilenet_v2.tflite`: Pre-trained classification model.
- `assets/imagenet_classes.txt`: Labels used for inference.
  
Set phone to USB debug model and connect with device.
Sync, build and Run.

## **4. Troubleshooting Experience**

### **4.1 Handling Dependency Conflicts in Python**
**Issue:**
TensorFlow and PyTorch dependencies caused conflicts with `typing-extensions`.

**Solution:**
Split dependencies into separate files and install them sequentially.
```sh
pip install -r pytorch_requirements.txt  # Install PyTorch dependencies
pip install -r tensorflow_requirements.txt  # Install TensorFlow dependencies
pip install -r typing_extensions_requirements.txt  # Fix conflicts
```

### **4.2 Fixing Android Forecasting Issues**
**Issue:**
The same image predicted **correctly in Python**, but **incorrectly in Android**.

![Model Design-2025-03-07-053916](https://github.com/user-attachments/assets/75cf00e7-b4e9-4dab-94f3-c8cd004876ee)

**Troubleshooting Process:**
1. Verified that the **TFLite model outputs correctly in Python**.
2. Compared **input values before inference in Python and Android**.
3. Found that **channel ordering was incorrect in Android**.

| **Step** | **Python Implementation** | **Android Implementation** | **Match?** |
| --- | --- | --- | --- |
| **Image Resizing** | `image.resize((224, 224), Image.BILINEAR)` | `Bitmap.createScaledBitmap(bitmap, 224, 224, true)` | ✅ Yes |
| **Convert Float** | `np.array(image, dtype=np.float32) / 255.0` | `((pixel shr X and 0xFF) / 255.0f)` | ✅ Yes |
| **Standardization** | `(image - mean) / std` | `((pixel_component / 255.0f) - mean) / std` | ✅ Yes |
| **Reorder Channels** | `np.transpose(image, (2, 0, 1))` | **❌ Not implemented in Android** | ❌ No |

**Solution:**
Explicitly reorder channels from HWC → CHW format in Android before passing input to the model.

## **5. Personal Blog**
For a detailed step-by-step breakdown, visit: wenyang.xyz/article/mlAndroid

## **6. Next Step**
The Android app's loading speed is reasonable but has noticeable delays, leaving room for improvement. 

