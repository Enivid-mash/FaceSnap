import numpy as np
from sklearn.svm import SVC
import pickle
import cv2

# Load pre-trained models and mean pre-processing data
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))

print('Models loaded successfully')

# Settings
gender_labels = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX

def pipeline_model(image_path, output_filename, color='bgr'):
    """
    Processes an image, detects the face, normalizes it, and predicts gender
    using a pre-trained SVM model and PCA transformation.

    Parameters:
    image_path (str): Path to the input image.
    output_filename (str): Filename to save the processed image with prediction.
    color (str): The color format of the image ('rgb' or 'bgr').

    Returns:
    None: The function saves the processed image to the specified path.
    """
    # Step-1: Read the image using OpenCV
    img = cv2.imread(image_path)

    # Step-2: Convert the image to grayscale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step-3: Detect faces using the Haar Cascade classifier
    faces = haar.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    if len(faces) == 0:
        # No face detected, display a message on the image
        message = "No face detected in the image"
        cv2.putText(img, message, (20, 50), font, 1, (0, 0, 255), 2)
    else:
        # If faces are detected, proceed with processing each face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Step-4: Crop the face region
            face_region = gray[y:y+h, x:x+w]

            # Step-5: Normalize the face region (scale values between 0 and 1)
            face_normalized = face_region / 255.0

            # Step-6: Resize the face to (100, 100)
            if face_normalized.shape[1] > 100:
                face_resized = cv2.resize(face_normalized, (100, 100), cv2.INTER_AREA)
            else:
                face_resized = cv2.resize(face_normalized, (100, 100), cv2.INTER_CUBIC)

            # Step-7: Flatten the resized face to a 1x10000 vector
            face_flattened = face_resized.reshape(1, 10000)

            # Step-8: Subtract the mean from the flattened face
            face_mean_adjusted = face_flattened - mean

            # Step-9: Transform the face using PCA to get the eigenface
            eigen_face = model_pca.transform(face_mean_adjusted)

            # Step-10: Use the pre-trained SVM model to predict the gender
            results = model_svm.predict_proba(eigen_face)[0]

            # Step-11: Extract the predicted gender and confidence score
            predicted_gender_index = results.argmax()
            confidence_score = results[predicted_gender_index]

            # Step-12: Create the text label with the predicted gender and score
            prediction_text = "%s : %0.2f" % (gender_labels[predicted_gender_index], confidence_score)
            cv2.putText(img, prediction_text, (x, y - 10), font, 1, (255, 255, 0), 2)

    # Save the processed image with prediction or message
    output_path = f'./static/predict/{output_filename}'
    cv2.imwrite(output_path, img)
