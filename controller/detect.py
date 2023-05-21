# import cv2
# import numpy as np
#
# app = Flask(__name__)
#
#
# @app.route('/')
# def home():
#     return render_template('../view/dashboard.html')
#
#
# @app.route('/detect', methods=['POST'])
# def detect():
#     # Get the uploaded image from the request
#     image = request.files['image']
#
#     # Save the image to a temporary file
#     image_path = 'temp.jpg'
#     image.save(image_path)
