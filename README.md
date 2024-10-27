# FaceSnap: AI-Powered Face Recognition Web Application

Welcome to the **FaceSnap** repository! This project is an AI-powered face recognition web application built with Python, OpenCV, Machine Learning, and Flask. It allows users to upload images and receive real-time face recognition results. The application leverages machine learning models like PCA (Eigenfaces) and Support Vector Machines (SVM) for accurate face classification.

![View of the landingpage](/static/homepage.PNG)
![View of the homepage](/static/app.PNG)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

**FaceSnap** is a web application designed to detect and recognize faces from uploaded images or in real-time. It integrates the power of OpenCV for image processing, PCA for dimensionality reduction, and SVM for face classification. The user-friendly interface, built with Flask, allows users to easily interact with the app and receive immediate results.

This project was developed as part of a Webstack Portfolio to demonstrate skills in machine learning, image processing, and web development.

## Features

- **Real-time Face Recognition:** Upload an image and get instant results.
- **PCA and SVM Integration:** Uses Eigenfaces (PCA) for feature extraction and SVM for classification.
- **Flask Web Interface:** Simple, clean, and user-friendly web interface for interacting with the recognition model.
- **Robust Error Handling:** Proper error handling for incorrect image uploads and system failures.
- **Visualization of Results:** Displays the recognized faces with bounding boxes and names.

## Technologies

- **Python**: Main programming language for backend and machine learning logic.
- **Flask**: Web framework used to develop the application interface.
- **OpenCV**: Library for image processing and face detection.
- **Scikit-learn**: Machine learning library for PCA (Principal Component Analysis) and SVM (Support Vector Machines).
- **HTML/CSS**: For front-end development and user interface design.

## Installation (Local Flask App)

To set up FaceSnap locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/FaceSnap.git
   cd FaceSnap
   ```

2. Set up a virtual environment:
    ```bash
    python3.8 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
   	python3 main.py 
    ```

## Usage
1. Navigate to `http://127.0.0.1:50000/` in your web browser.
2. Upload Image: On the App page, click the "Choose File" button to upload an image containing faces.
3. Recognition: Once uploaded, the app will process the image, detect faces, and display the recognized faces along with their bounding boxes and labels
4. View Results: The results will be displayed directly on the web page, with options to upload new images.
