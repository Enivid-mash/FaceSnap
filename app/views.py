from flask import render_template, request
from flask import redirect, url_for
import os
from PIL import Image
from app.utils import pipeline_model

UPLOAD_FOLDER = 'static/uploads'

def base():
    return render_template('base.html')


def index():
    return render_template('index.html')


def faceapp():
    return render_template('faceapp.html')


def get_image_width(path):
    """
    Get the width of the image to maintain aspect ratio for display.

    Parameters:
    path (str): Path to the image file.

    Returns:
    int: The width of the image to maintain a proper aspect ratio.
    """
    img = Image.open(path)
    size = img.size  # (width, height)
    aspect_ratio = size[0] / size[1]  # width / height
    width = int(300 * aspect_ratio)  # Scaling height to 300 and calculating width
    return width


def gender():
    """
    Handles the file upload for gender prediction and renders the result.

    Returns:
    Flask response: Renders the 'gender.html' template with the prediction results.
    """
    if request.method == "POST":
        # Get the uploaded image file
        uploaded_file = request.files['image']
        filename = uploaded_file.filename
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the uploaded file to the server
        uploaded_file.save(upload_path)
        
        # Get the width of the uploaded image to maintain aspect ratio
        img_width = get_image_width(upload_path)
        
        # Call the pipeline model for gender prediction and pass the image path
        pipeline_model(upload_path, filename, color='bgr')
        
        # Render the result page with the uploaded image and prediction
        return render_template('gender.html', fileupload=True, img_name=filename, w=img_width)

    # If the request method is not POST, render the default page
    return render_template('gender.html', fileupload=False, img_name="freeai.png")
