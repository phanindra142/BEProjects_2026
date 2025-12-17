Team 6 - BE Project 2026
# 📘 ReadCam – Standalone Realtime Book Reading Assistant

ReadCam is a **standalone, offline, IoT-based assistive reading device** designed to help **visually impaired, elderly, and reading-challenged users** convert printed text into clear, real-time speech. Built using **Raspberry Pi 4B+**, **Pi Camera Module 3**, and open-source AI tools, ReadCam performs **image capture, OCR, speech recognition, and text-to-speech locally**, without any internet dependency.

---

## 🚀 Key Features

- 📷 **Real-time Image Capture** using Raspberry Pi Camera Module 3  
- 🧠 **Offline OCR** using Tesseract with OpenCV preprocessing  
- 🎙️ **Offline Voice Commands** using VOSK Speech Recognition  
- 🔊 **Offline Text-to-Speech** using PicoTTS  
- 🔁 **Dual Operation Modes**
  - **Automatic Mode** (continuous reading)
  - **Manual Mode** (voice-controlled: capture, read, speak, stop)
- 🔘 **GPIO Hardware Switch** to toggle modes
- 🌐 **Fully Offline & Privacy-Preserving**
- 💰 **Low-Cost & Portable Assistive Technology**

---

## 🎯 Problem Addressed

Existing reading solutions often:
- Depend on smartphones and internet connectivity
- Are expensive (commercial assistive devices)
- Have complex user interfaces
- Perform poorly in real-world conditions

**ReadCam solves these issues** by offering a **simple, affordable, standalone, and offline reading assistant** suitable for rural areas, schools, libraries, and homes.

---

## 🛠️ Hardware Requirements

| Component | Description |
|---------|-------------|
| Raspberry Pi 4B+ | Main processing unit |
| Raspberry Pi Camera Module 3 | Image capture |
| USB Microphone | Voice command input |
| USB / 3.5mm Speaker | Audio output |
| GPIO Push Button / Switch | Mode selection |
| microSD Card (32GB+) | OS and storage |
| 5V/3A Power Supply | Stable operation |

---

## 💻 Software Stack

- **OS:** Raspberry Pi OS  
- **Language:** Python 3.9+  
- **Libraries & Tools:**
  - OpenCV – image preprocessing
  - Tesseract OCR – text extraction
  - VOSK – offline speech recognition
  - PicoTTS (`pico2wave`) – text-to-speech
  - Picamera2 – camera handling
  - RPi.GPIO – hardware interaction
  - sounddevice, numpy, threading

---


All modules run using **multithreading** to ensure smooth real-time performance.

---

## 🗣️ Supported Voice Commands

| Command | Action |
|-------|--------|
| `capture` | Capture image |
| `read` / `speak` | Read extracted text |
| `stop` | Stop reading |
| `exit` | Terminate program |

---

## ⚙️ How It Works

1. User places a printed document under the camera  
2. Device operates in **Automatic** or **Manual (Voice)** mode  
3. Image is captured and preprocessed  
4. Text is extracted using OCR  
5. Extracted text is spoken aloud  
6. System waits for next command or continues automatically  

---

## 📊 Performance Highlights

- OCR Accuracy: **~89% – 94%** (normal lighting)
- Average End-to-End Latency: **~2 seconds**
- Voice command recognition works reliably in moderate noise
- Stable long-duration operation with low CPU usage

---

## 📈 Applications

- Assistive reading for visually impaired users  
- Elderly-friendly reading aid  
- Rural schools and libraries  
- Hospitals and public service centers  
- Offline learning environments  

---

## 🔮 Future Enhancements

- 🌍 Multilingual OCR & TTS (Hindi, Kannada, Tamil, etc.)
- ✍️ Handwritten text recognition (CRNN / ViT models)
- 🔋 Battery-powered portable version
- 📱 Mobile companion application
- 🕶️ Integration with smart glasses
- 🎚️ Custom voice speed and tone settings
- 📳 Haptic feedback for accessibility

---

## 📚 Project Domain

- Assistive Technology  
- Embedded Systems  
- Internet of Things (IoT)  
- Computer Vision  
- Edge AI  

---


