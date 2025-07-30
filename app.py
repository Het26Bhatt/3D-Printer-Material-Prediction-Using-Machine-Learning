import os
import pickle
from flask import Flask, request, render_template
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading ---
# Define the path to your model.pkl file.
# This assumes 'model.pkl' is in the same directory as 'app.py' or a specified subfolder.
# Based on your previous command prompt output, if app.py is run from 'C:\Users\Swedan\Desktop\LP3 - VS_New\'
# and your model is in 'C:\Users\Swedan\Desktop\LP3 - VS_New\LP3 VS\model.pkl',
# then the path should be relative like 'LP3 VS/model.pkl'.
# ADJUST THIS PATH if your model.pkl is located differently.
model_path = 'model.pkl' # <-- VERIFY THIS PATH CAREFULLY

try:
    # Load the pre-trained machine learning model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model '{model_path}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Please ensure it's in the correct directory.")
    model = None # Set model to None to prevent further errors if not loaded
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Route for the Home Page ---
# This route handles GET requests to the root URL (e.g., http://127.0.0.1:5000/)
@app.route('/')
def home():
    # Render the index.html template.
    return render_template('index.html')

# --- Route for Prediction ---
# This route handles POST requests, typically from a form submission.
@app.route('/predict', methods=['POST'])
def predict():
    prediction_message = "An unexpected error occurred. Please try again." # Default error message

    if model is None:
        prediction_message = "Error: The prediction model could not be loaded. Please contact support."
        return render_template('index.html', prediction_text=prediction_message)

    try:
        # Get the input data from the HTML form.
        x_direction = float(request.form['x-direction'])
        y_direction = float(request.form['y-direction'])
        z_direction = float(request.form['z-direction'])

        # Create a NumPy array from the input data.
        input_data = np.array([[x_direction, y_direction, z_direction]])

        # Make the prediction using the loaded model.
        prediction = model.predict(input_data)[0]

        # Generate the appropriate message based on the prediction result.
        if prediction == 1:
            prediction_message = "Yes, There might be a problem in the 3D Printer. Please check."
        else:
            prediction_message = "No, There is no problem in the 3D Printer. You can continue working."

    except KeyError as e:
        prediction_message = f"Error: Missing form data. Please ensure all fields are filled. Missing: {e}"
        print(f"KeyError: {e} - Ensure HTML form field names match Python keys.")
    except ValueError:
        prediction_message = "Error: Invalid input. Please enter numeric values for all fields."
        print("ValueError: Non-numeric input received.")
    except Exception as e:
        prediction_message = f"An error occurred during prediction: {e}"
        print(f"Prediction Error: {e}")

    # Render the index.html template again, passing the prediction message
    return render_template('index.html', prediction_text=prediction_message)

# --- Run the Flask Application ---
# This ensures the Flask development server runs only when the script is executed directly.
# debug=True allows for automatic reloading on code changes and provides a debugger.
# IMPORTANT: Set debug=False in a production environment.
if __name__ == '__main__': # <--- THIS LINE IS CRUCIAL
    app.run(debug=True) # <--- THIS LINE IS CRUCIAL