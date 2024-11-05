from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from plyer import filechooser as fc
import tensorflow as tf  # Import TensorFlow
import numpy as np

class MainLayout(BoxLayout):
    def upload_original_signature(self):
        file_path = fc.open_file(title="Select Original Signature")
        if file_path:
            self.ids.original_image.source = file_path[0]
            self.ids.original_image.reload()

    def upload_test_signature(self):
        file_path = fc.open_file(title="Select Test Signature")
        if file_path:
            self.ids.test_image.source = file_path[0]
            self.ids.test_image.reload()

    def verify_signatures(self):
        # Load the signature verification model
        model = tf.keras.models.load_model('signature_verification_model.h5')

        # Preprocess images (resize, grayscale, etc.) - Implement your preprocessing logic here
        original_image = preprocess_image(self.ids.original_image.source)
        test_image = preprocess_image(self.ids.test_image.source)

        # Make predictions using the model
        prediction = model.predict([original_image, test_image])

        # Update progress bar (optional)
        # self.progress_bar.value = 100

        # Update result label based on prediction
        result = "Match" if prediction[0][0] > 0.5 else "Mismatch"
        self.ids.result_label.text = f"Verification Result: {result}"

def preprocess_image(image_path):
    # Read the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))  # Adjust target size as needed

    # Convert to grayscale
    image = tf.keras.preprocessing.image.img_to_array(image, dtype='uint8')
    image = tf.image.rgb_to_grayscale(image)

    # Normalize pixel values to the range [0, 1]
    image = image / 255.0

    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)

    return image

class AuthentechApp(App):
    def build(self):
        return MainLayout()

app = AuthentechApp()
app.run()