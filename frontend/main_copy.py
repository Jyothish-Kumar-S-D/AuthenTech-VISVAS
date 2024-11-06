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
from pathlib import Path
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import preprocessing

class MainLayout(BoxLayout):
    pass

class AuthentechApp(MDApp):
    progress_value = NumericProperty(0)
    dialog = None
    result_text = " "
    
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file('authentech.kv')
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image=preprocessing.preprocess(image)
        return np.expand_dims(image, axis=0)
    
    def upload_original_signature(self):
        file_path = fc.open_file(
            title="Select Original Signature",
            filters=[("Image Files", "*.png", "*.jpg", "*.jpeg")]
        )
        if file_path:
            file_path = file_path[0]
            self.root.ids.original_image.source = file_path
            self.root.ids.original_image.reload()

    def upload_test_signature(self):
        file_path = fc.open_file(
            title="Select Test Signature",
            filters=[("Image Files", "*.png", "*.jpg", "*.jpeg")]
        )
        if file_path:
            file_path = file_path[0]
            self.root.ids.test_image.source = file_path
            self.root.ids.test_image.reload()

    def start_verification(self):
        if not self.root.ids.original_image.source or not self.root.ids.test_image.source:
            self.show_dialog("Please upload both signatures first!")
            return
        self.root.ids.progress_bar.value = 0
        self.root.ids.verify_button.disabled = True
        Clock.schedule_interval(self.update_progress, 0.1)

    def update_progress(self, dt):
        if self.root.ids.progress_bar.value >= 100:
            self.root.ids.verify_button.disabled = False
            self.compare_signatures()
            return False
        self.root.ids.progress_bar.value += 2

    def compare_signatures(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signature_verification_model.h5')
        model=load_model(model_path)
        original_image = self.preprocess_image(self.root.ids.original_image.source)
        test_image = self.preprocess_image(self.root.ids.test_image.source)

        if original_image is None or test_image is None:
            self.show_dialog("Error processing images!")
            return
        
        prediction = model.predict([original_image, test_image])
        result = "The Test Signature is NOT AUTHENTIC!" if prediction[0][0] > 0.5 else "Signature Verified as AUTHENTIC"
        self.result_text = f"Verification Result: {result}"
        self.show_dialog(self.result_text)

    def show_dialog(self, message):
        if self.dialog is None:
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
        else:
            self.dialog.text = message  # Update the text for the dialog if it already exists
        self.dialog.open()


    def close_dialog(self, *args):
        if self.dialog:
            self.dialog.dismiss()

if __name__ == '__main__':
    AuthentechApp().run()
