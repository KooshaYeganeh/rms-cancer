import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler
# Image transformation for preprocessing
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, url_for , send_file , Response ,jsonify
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import io
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from skimage import color, filters, measure
from skimage import io as skio, color, filters, measure, morphology, segmentation
import matplotlib
import pickle
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import cv2







# Initialize Flask app
app = Flask(__name__)



upload_folder = os.path.join(f"uploads")
os.makedirs(upload_folder, exist_ok=True)


app.config.update(SECRET_KEY="rms-cancer")





login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"{self.id}"


# Sample hardcoded user data (username and plain text password)
users = {
    "sanaat": "sanaat_123123"  # Plain text password
}


# Load user from the ID
@login_manager.user_loader
def load_user(userid):
    return User(userid)


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.id)


@app.route("/login")
def login():
    return render_template("sign-in.html")


@app.route("/login", methods=["POST"])
def loggin():
    username = request.form["username"]
    password = request.form["password"]

    # Check if the username exists and password matches in plain text
    if username in users and users[username] == password:
        user = User(username)
        login_user(user)  # Log the user in
        return redirect(url_for("dashboard"))  # Redirect to the protected dashboard
    else:
        return render_template("sign-in.html", error="Username or password is invalid")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# Error handler for unauthorized access
@app.errorhandler(401)
def page_not_found(e):
    return Response("""
                    <html><center style='background-color:white;'>
                    <h2 style='color:red;'>Login failed</h2>
                    <h1>Error 401</h1>
                    </center></html>""")



""" This Route Get histopathologic_cancer Detection. Give Image Like Samples
to return Valid Response"""


@app.route("/mamo_cancer")
def breast_cancer():
    return render_template("mamo_cancer.html")


# Define the CNN model for cancer detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#class SimpleCNN_mamo(nn.Module):
class SimpleCNN_mamo(nn.Module):
    def __init__(self):
        super(SimpleCNN_mamo, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 2)  # Output for 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv + ReLU + Pooling
        x = x.view(-1, 16 * 112 * 112)  # Flatten the tensor
        x = self.fc1(x)  # Fully connected layer
        return x

# Initialize model and load pre-trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mamo_model = SimpleCNN_mamo().to(device)
mamo_model.load_state_dict(torch.load('mamo_classifier.pth', map_location=device))
mamo_model.eval()  # Set model to evaluation mode

# Class names for the prediction output
mamo_class_names = ['Abnormal', 'Normal']  # Adjust according to your dataset

# Define image transformations
mamo_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to the model's input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize tensor values
])

# Image Processing Function
def process_image_mamo(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    img = mamo_transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)

# Flask route to handle file upload and prediction
@app.route('/mamo_cancer', methods=['POST'])
def upload_and_predict_mamo():
    mamo_image = request.files['mamo_image']

    if mamo_image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded image temporarily to a file
        image_path = f"{upload_folder}/{mamo_image.filename}"
        mamo_image.save(image_path)

        # Process the image and transform it
        image = process_image_mamo(image_path)

        # Model prediction
        with torch.no_grad():
            outputs = mamo_model(image)
            _, predicted = torch.max(outputs, 1)

        # Map predicted index to class name
        predicted_class = mamo_class_names[predicted.item()]

        # Return the prediction result
        return render_template("mamo_cancer.html" , prediction = predicted_class)

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500



@app.route("/mamo_cancer_ultrason")
def mamo_cancer_ultrasound_get():
    return render_template("mamo_cancer_ultrason.html")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (should match the training script)
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained model
ultrason_model = ImprovedCNN().to(device)
ultrason_model.load_state_dict(torch.load('best_model_monai_breast_ultrason.pth', map_location=device))
ultrason_model.eval()  # Set model to evaluation mode

# Define transformation (must match the training script)
ultrason_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the route for image upload and prediction
@app.route('/predict_mamo_ultrason', methods=['POST'])
def predict_mamo_ultrason():
    file = request.files['mamo_ultrason']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Open the image file
        image = Image.open(file).convert('RGB')

        # Apply transformations
        image_tensor = ultrason_transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = ultrason_model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Class mapping
        classes = ['benign', 'malignant', 'normal']
        result = classes[predicted.item()]

        return render_template( "mamo_cancer_ultrason.html" , prediction = result )







@app.route('/lung_cancer')
def lung_cancer_get():
    return render_template('lung_cancer.html')



class SimpleCNN_lung(nn.Module):
    def __init__(self):
        super(SimpleCNN_lung, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 4)  # Output for 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = self.fc1(x)
        return x

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lung = SimpleCNN_lung().to(device)
model_lung.load_state_dict(torch.load('lung_cancer_monai_classifier.pth', map_location=device))
model_lung.eval()  # Set model to evaluation mode

# Define the class labels for lung cancer types
lung_classes = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Image preprocessing function
def preprocess_image_lung(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    # Resize to match model input size
    img = cv2.resize(img, (224, 224))
    
    # Convert to RGB, normalize, and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    img = transform(img)
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
    return img

# Prediction function
def predict_tumor_lung(image_path):
    processed_image = preprocess_image_lung(image_path)
    with torch.no_grad():
        outputs = model_lung(processed_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    # Get the predicted label
    predicted_label = lung_classes[predicted_class]
    
    # Compute confidence score using softmax
    softmax = torch.nn.Softmax(dim=1)
    confidence = softmax(outputs).max().item()
    
    return predicted_label, confidence

# Flask route to handle file upload and model prediction
@app.route('/lung_cancer_detection', methods=['POST'])
def lung_cancer_detection():
    # Get the uploaded file
    lungfile = request.files['lung_image']
    
    if lungfile.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file to a temporary location
    try:
        image_path = os.path.join(upload_folder, lungfile.filename)
        lungfile.save(image_path)

        # Predict the tumor type
        predicted_class, confidence = predict_tumor_lung(image_path)

        # Return the prediction result
        return render_template("lung_cancer.html" , prediction =  predicted_class)
    except Exception as e:
        return jsonify({"error": str(e)}), 500







@app.route("/kidney_disease")
def kidney_disease():
    return render_template("kidney_disease.html")


class SimpleCNN_kidney(nn.Module):
    def __init__(self):
        super(SimpleCNN_kidney, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 4)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc1(x)
        return x

# Load model
kidney_model = SimpleCNN_kidney().to(device)
kidney_model.load_state_dict(torch.load('kidney_classifier.pth', map_location=device))
kidney_model.eval()

# Define class names for kidney disease
kidney_class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Image transformations
kidney_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Image processing function
def process_image_kidney(image_path):
    img = Image.open(image_path).convert('RGB')
    img = kidney_transform(img)
    img = img.unsqueeze(0)
    return img.to(device)




@app.route('/kidney_disease', methods=['POST'])
def predict_kidney_disease():
   
    file = request.files['kidney_image']
    if file.filename == '':
        return "No selected file"
    
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Process the image
        img = process_image_kidney(file_path)

        # Predict with the model
        with torch.no_grad():
            output = kidney_model(img)
            _, predicted = torch.max(output, 1)
            predicted_class = kidney_class_names[predicted.item()]

        # Return prediction result
        return render_template('kidney_disease.html', prediction = predicted_class)

    except Exception as e:
        return str(e)







@app.route("/brain_tumor")
def brain_tumor_detection():
    return render_template("brain_tumor_classifier.html")



# Device configuration (for GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture for brain tumor classification
class SimpleCNN_brain(nn.Module):
    def __init__(self):
        super(SimpleCNN_brain, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 4)  # Output for 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = self.fc1(x)
        return x

# Load the pre-trained model for brain tumor classification
model_brain = SimpleCNN_brain().to(device)
model_brain.load_state_dict(torch.load('monai_brain_classifier.pth', map_location=device))
model_brain.eval()  # Set model to evaluation mode

# Define class labels for brain tumor types
brain_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image preprocessing for brain tumor
def preprocess_image_brain(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    # Resize to match model input size
    img = cv2.resize(img, (224, 224))
    
    # Convert to RGB, normalize, and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    img = transform(img)
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
    return img

# Model prediction for brain tumor
def predict_tumor_brain(image_path):
    processed_image = preprocess_image_brain(image_path)
    with torch.no_grad():
        outputs = model_brain(processed_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    predicted_label = brain_classes[predicted_class]
    
    # Compute confidence score using softmax
    softmax = torch.nn.Softmax(dim=1)
    confidence = softmax(outputs).max().item()
    
    return predicted_label, confidence

# Flask route to handle the image upload and prediction
@app.route("/brain_tumor", methods=["GET", "POST"])
def brain_tumor_classifier():
    if request.method == "POST":        
        file = request.files['brain_image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        try:
            # Save the uploaded file to a temporary folder
            image_path = os.path.join(upload_folder, file.filename)
            file.save(image_path)

            # Predict the tumor class and confidence
            predicted_class, confidence = predict_tumor_brain(image_path)

            # Return the prediction as a JSON response
            return render_template("brain_tumor_classifier.html" , prediction = predicted_class)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # If GET request, just render the form for uploading
        return render_template("brain_tumor_classifier.html")






@app.route("/leukemia_classifier")
def leukemia_classifier_get():
    return render_template("leukemia_classifier.html")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
class SimpleCNN_Leukemia(nn.Module):
    def __init__(self):
        super(SimpleCNN_Leukemia, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



""" 
Layers:

    conv1: A convolutional layer that applies a 3x3 kernel to the RGB image, outputting 16 feature maps.
    pool: A max-pooling layer that reduces the dimensions by half.
    fc1: A fully connected layer that maps to the four ALL classes.

Forward Pass:

    The image is passed through the convolutional and pooling layers, then flattened and classified by the fully connected layer.

"""


leukemia_model = SimpleCNN_Leukemia().to(device)
leukemia_model.load_state_dict(torch.load('leukemia_classifier.pth', map_location=device))
leukemia_model.eval()


""" 
Loads pre-trained weights from the file leukemia_classifier.pth and sets the model to evaluation mode.
"""

# Define class names
leukemia_model_class_names = ['Benign', 'Early', 'Pre', 'Pro']  # The output classes represent the progression stages of ALL.

""" 
Resize: Rescales the image to 224x224 pixels.
ToTensor: Converts the image to a PyTorch tensor.
Normalize: Standardizes the pixel values to speed up training and improve model performance.
"""

# Define image transformations
leukemia_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])




""" 
Loads and preprocesses the image, adding a batch dimension and sending it to the specified device.
"""
# Helper function to process image
def process_image_leukemia(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    img = leukemia_transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)




""" 
File Handling: Checks for a valid image file upload, saves it temporarily, and processes it.
Inference: Preprocesses the image, runs it through the model, and retrieves the predicted class.
Cleanup: Removes the uploaded image from the server after prediction.
Output: Returns the predicted class on a result page, or an error message in JSON format if an exception occurs.
"""

# Define the prediction route
@app.route('/monai/predict_leukemia', methods=['POST'])
def predict_blood_cell():
    file = request.files['leuimage']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file temporarily
        image_path = os.path.join(upload_folder , file.filename)  # Ensure this directory exists
        file.save(image_path)

        # Preprocess the image and perform inference
        try:
            img = process_image_leukemia(image_path)
            with torch.no_grad():
                output = leukemia_model(img)
                _, predicted = torch.max(output, 1)
                predicted_class = leukemia_model_class_names[predicted.item()]

            # Clean up: remove the uploaded image after processing
            os.remove(image_path)

            # Return the prediction as JSON or render a result page
            return render_template("leukemia_classifier.html", prediction = predicted_class)

        except Exception as e:
            return jsonify({'error': str(e)}), 500





@app.route("/brest_cancer_ml")
def breast_cancer_detection_with_machine_learning():
    return render_template("breast_cancer_ml.html")





# Load the trained model and label encoder
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('breast_label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Route for home page with form


# Route for making predictions with form data
@app.route('/brest_cancer_ml', methods=['POST'])
def brest_cancer_ml_post():
        # Extract the data from the form fields
    features = [
        float(request.form['radius_mean']),
        float(request.form['texture_mean']),
        float(request.form['perimeter_mean']),
        float(request.form['area_mean']),
        float(request.form['smoothness_mean']),
        float(request.form['compactness_mean']),
        float(request.form['concavity_mean']),
        float(request.form['concave_points_mean']),
        float(request.form['symmetry_mean']),
        float(request.form['fractal_dimension_mean']),
        float(request.form['radius_se']),
        float(request.form['texture_se']),
        float(request.form['perimeter_se']),
        float(request.form['area_se']),
        float(request.form['smoothness_se']),
        float(request.form['compactness_se']),
        float(request.form['concavity_se']),
        float(request.form['concave_points_se']),
        float(request.form['symmetry_se']),
        float(request.form['fractal_dimension_se']),
        float(request.form['radius_worst']),
        float(request.form['texture_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['area_worst']),
        float(request.form['smoothness_worst']),
        float(request.form['compactness_worst']),
        float(request.form['concavity_worst']),
        float(request.form['concave_points_worst']),
        float(request.form['symmetry_worst']),
        float(request.form['fractal_dimension_worst']),
    ]
    
    # Convert the form data into a DataFrame (for compatibility with the model)
    input_data = pd.DataFrame([features])

    # Make the prediction
    prediction = model.predict(input_data)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]
    if decoded_prediction == "B":
        prediction_result = "Benign"
    elif decoded_prediction == "M":
        prediction_result = "Malignant"
    else:
        return "Not Detected"

    # Render the prediction result back to the user
    return render_template('breast_cancer_ml.html', result = prediction_result)





@app.route("/about_us")
def about_us():
    return render_template("about_us.html")





@app.route("/tables")
def tables_view():
    return render_template("tables.html")


if __name__ == '__main__':
    app.run(debug=True , port = 5005)
