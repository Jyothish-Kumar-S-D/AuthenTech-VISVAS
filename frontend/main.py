from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from plyer import filechooser as fc
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.clock import Clock
from kivy.properties import NumericProperty
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

class MainLayout(BoxLayout):
    pass

class AuthentechApp(MDApp):
    progress_value = NumericProperty(0)
    dialog = None
    result_text = " "
    # original_array = None  # Store the numpy array for original signature
    # test_array = None      # Store the numpy array for test signature

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file('authentech.kv')

    # def image_to_numpy(self, image_path, size=(224, 224)):

    #     image = cv2.imread(image_path)
    #     if image is None:
    #         self.show_dialog("Error loading image!")
    #         return None
            
    #     # Convert to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    #     # Resize to consistent size
    #     resized = cv2.resize(gray, size)
        
    #     # Normalize pixel values
    #     normalized = resized / 255.0
        
    #     return normalized
    
    def preprocess_image(self, image_array, target_size=(224, 224)):
        # Resize the image
        image = tf.keras.preprocessing.image.array_to_img(image_array)
        image = image.resize(target_size)
        
        # Convert to array and preprocess
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array

    def upload_original_signature(self):
        file_path = fc.open_file(
            title="Select Original Signature",
            filters=[("Image Files", "*.png", "*.jpg", "*.jpeg")]
        )
        if file_path:
            file_path = file_path[0]
            # Update the image in the UI
            self.root.ids.original_image.source = file_path
            self.root.ids.original_image.reload()

    def upload_test_signature(self):
        file_path = fc.open_file(
            title="Select Test Signature",
            filters=[("Image Files", "*.png", "*.jpg", "*.jpeg")]
        )
        if file_path:
            file_path = file_path[0]
            # Update the image in the UI
            self.root.ids.test_image.source = file_path
            self.root.ids.test_image.reload()

    def start_verification(self):
        if self.root.ids.original_image.source is None or self.root.ids.test_image.source is None:
            self.show_dialog("Please upload both signatures first!")
            return
            
        self.root.ids.progress_bar.value = 0
        self.root.ids.verify_button.disabled = True
        Clock.schedule_interval(self.update_progress, 0.1)

    def update_progress(self, dt):
        if self.root.ids.progress_bar.value >= 100:
            self.root.ids.verify_button.disabled = False

            self.show_dialog(self.result_text)
            return False
        self.root.ids.progress_bar.value += 2

    def compare_signatures(self):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the full path to the model file
        model_path = os.path.join(script_dir, 'signature_verification_model.h5')
        
        if os.path.exists(model_path):
            # Load the signature verification model
            model = load_model(model_path)
            
            # Preprocess the original and test images
            original_image = self.preprocess_image(self.root.ids.original_image.source)
            test_image = self.preprocess_image(self.root.ids.test_image.source)
            
            # Make predictions using the model
            prediction = model.predict([original_image, test_image])
            
            # Update result label based on prediction
            result = "Match" if prediction[0][0] > 0.5 else "Mismatch"
            self.result_text = f"Verification Result: {result}"
        else:
            self.show_dialog(f"Error: Unable to find the model file at '{model_path}'")

    def show_dialog(self, message):
        if not self.dialog:
            self.dialog = MDDialog(
                title="Notice",
                text=message,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=self.close_dialog
                    )
                ]
            )
        self.dialog.open()

    def close_dialog(self, *args):
        self.dialog.dismiss()

if __name__ == '__main__':
    AuthentechApp().run()