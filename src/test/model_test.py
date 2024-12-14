import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def model_test(model, class_names):
    # List all image files in the specified folder
    folder_path = 'Data/origin/real_test_set'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        print(f"Processing: {image_file}")
        
        # Load the image and preprocess it
        image_path = os.path.join(folder_path, image_file)
        test_image = load_img(image_path, target_size=(124, 124))
        
        # Display the image
        plt.imshow(test_image, interpolation='spline16')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        
        # Convert the image to a numpy array and add batch dimension
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Get predictions
        result = model.predict(test_image)
        
        # Print predictions
        for i, label in enumerate(class_names):
            print(f"\t{label} ==> {result[0][i] * 100:.2f} %")
        print("\n")  # Separate output for each image

