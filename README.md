# Real-Time Speaker Spotlight Using Linear Microphone Array 🎙️👁️

An intelligent, autonomous hardware-software integrated edge computing system developed for **CS724: Sensing Communications and Networking for Smart Wireless Devices** at the **Indian Institute of Technology, Kanpur (IIT Kanpur)**. 

The system localizes an active sound source using digital signal processing algorithms over a multi-channel linear microphone array, cross-verifies speech via human face lip mesh movements, and locks focus onto speakers without human intervention.

---

## 👥 Project Team (Group 8)
* Pokala Dattatreya, Edula Vinay Kumar Reddy, Kethavath Srinu, Voora Nagendra
* Vikram Kumar, Vijiyant Tanaji Shejwalkar, Yashwanth Tippireddy, Telugu Sudhakar
* **Under the Guidance of:** Prof. Amitangshu Pal (IIT Kanpur)

---

## 📊 Core System Capabilities & Results
* **High-Accuracy Audio Localization:** Achieved **90% accuracy** in detecting a speaker's precise angular position within an error margin of **±10°**.
* **Enhanced Visual Tracking:** Migrating from YOLO to **MediaPipe Face Mesh** provided a **20% tracking efficiency gain**, ensuring pinpoint landmark matching.
* **Ultra-Low Latency Implementation:** The entire system architecture handles capture-to-servo correction pipelines in **under 0.1 seconds**.

---

## 🏗️ Hardware Architecture Components
The hardware processing structure is composed of:
1. **Raspberry Pi Single-Board Computer** - Core calculation hub processing pipeline operations.
2. **ReSpeaker 4-Mic Linear Array** - Configured with an exact 5 cm linear inter-element microphone spacing layout.
3. **Raspberry Pi Camera Module v2** - Weight-optimized capture module mounted directly atop a rotating assembly.
4. **TowerPro 180° Position-Feedback Servo Motor** - Adjusts camera physical yaw vector relative to sound directions.
5. **GPIO 1-to-3 Expansion Board** - Separates physical signal distribution pins.

---

## 🔬 Computational Signal Processing Pipeline

### 1. Sound Direction of Arrival (DOA) Estimation
* **Sampling Rate:** Audio streams enter at 16,000 Hz with frame chunk buffers restricted to 1024 points.
* **Frequency Analysis:** Fast Fourier Transforms (FFT) map streaming data arrays into frequency scopes.
* **Cross-Power Spectrum Phase:** Computes precise time-delay metrics between the physical microphone lines using Phase Cross Correlation, which is converted to an exact angular location vector via an Inverse FFT (IFFT).
* **Signal Filtering:** Employs Root Mean Square (RMS) filtering parameters to automatically reject ambient acoustic noise profiles.

### 2. Multi-User Visual Tracking & Validation
* MediaPipe processes live video matrices to match facial landmarks.
* Measures the changing Euclidean distance boundaries between inner lip points to evaluate the speech status of the user.
* Combines the audio DOA vector and lip-movement status to highlight the target speaker.

---

## 🛠️ Step-by-Step Installation Guide

### 1. Provisioning Raspberry Pi OS Linux Dependencies
Execute the hardware audio setup configuration script via your terminal terminal:
```bash
chmod +x config/raspi_setup.sh
./config/raspi_setup.sh
```

### 2. Setting Up Python Components
Install the required application stack imports:
```bash
pip install -r requirements.txt
```

### 3. Execution
Launch the core integrated monitoring engine:
```bash
python src/app.py
```
Open a browser page on any device connected to the local area network network and access the web video interface link: `http://localhost:5000`.
