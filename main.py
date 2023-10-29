import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # Import Image and ImageTk from Pillow
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model


def compare_images(input_image, image_folder_path):
    input_img = cv2.imread(input_image, cv2.IMREAD_COLOR)

    if input_img is None:
        return "Input image not found."

    for root, dirs, _ in os.walk(image_folder_path):
        for folder_name in dirs:
            folder_path = os.path.join(image_folder_path, folder_name)
            for filename in os.listdir(folder_path):
                folder_img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_COLOR)
                if folder_img is not None:
                    # Ensure the input and folder images have the same size
                    folder_img = cv2.resize(folder_img, (input_img.shape[1], input_img.shape[0]))

                    # Calculate the absolute difference and mean
                    abs_diff = cv2.absdiff(input_img, folder_img)
                    diff_mean = abs_diff.mean()

                    if diff_mean == 0:
                        return f"Disease found as: {folder_name}"
    return "No matching image found in any folder."


def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the selected image
        input_image = Image.open(file_path)
        input_image.thumbnail((200, 200))  # Resize the image for the preview
        input_image = ImageTk.PhotoImage(input_image)
        
        # Display the selected image in the preview label
        input_image_label.config(image=input_image)
        input_image_label.image = input_image  # Keep a reference to prevent garbage collection

        # Perform image comparison and display the result
        result = compare_images(file_path, image_folder_path)
        result_label.config(text=result)

        # Schedule the next update after a certain time interval (e.g., 500 milliseconds)
        root.after(500, browse_image)

def create_gui(background_image_path):
    # Create a Tkinter window
    # Load the pre-trained VGG16 model without the top (classification) layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Modify the model for your specific classification task
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)  # Add your own dense layers as needed
    predictions = tf.keras.layers.Dense(4, activation='softmax')(x)  # Replace NUM_CLASSES with the number of skin conditions

    model = Model(inputs=base_model.input, outputs=predictions)
    root = tk.Tk()
    root.title("Skincare A.I.")

    # Load the background image
    original_bg = Image.open(background_image_path)
    bg_image = ImageTk.PhotoImage(original_bg)

    # Set the background image
    background_label = tk.Label(root, image=bg_image)
    background_label.place(relwidth=1, relheight=1)

    # Increase the size of the GUI window
    root.geometry("800x600")  # Adjusted size for the image preview

    # Create and configure message label
    message_label = tk.Label(root, text="Select an image to enter", font=("Arial", 14))
    message_label.pack(pady=20)

    # Create a frame for the image preview
    image_preview_frame = tk.Frame(root)
    image_preview_frame.pack(pady=10)
    
    # Create a label to display the selected image
    global input_image_label
    input_image_label = tk.Label(image_preview_frame)
    input_image_label.pack(side="left")

    # Create a button to browse for an image
    browse_button = tk.Button(root, text="Browse Image", command=browse_image, font=("Arial", 12))
    browse_button.pack(pady=10)

    # Create a label to display the result
    global result_label
    result_label = tk.Label(root, text="", font=("Arial", 14))
    result_label.pack(pady=20)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    image_folder_path = "images"  # The folder containing subfolders with images
    background_image_path = "C:\\Users\\Ishan\\Desktop\\img rec\\doctor2.png"  # Use the path to your background image (e.g., .jpg, .png)

    create_gui(background_image_path)