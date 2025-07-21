import os
from PIL import Image

def resize_and_save_images(data_name, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    for i, image_file in enumerate(image_files, start=1):
        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path)

        if img.mode in ['P', 'RGBA']:
            img = img.convert('RGB')

        img_resized = img.resize((224, 224))
        new_name = f"{i:03d}.jpg"

        # Save the adjusted image
        output_path = os.path.join(output_folder, new_name)
        img_resized.save(output_path)

        print(f"Image preprocessing in progress, saved: {new_name}")


# Main path configuration
our_dataset = "Concrete-Image-Classification-main/our_dataset/"
categories = ["normal", "honeycomb", "hole", "crack", "corrosion_of_reinforcement"]

for category in categories:
    folder_path = f'{our_dataset}origin/{category}'

    output_folder_train = f'{our_dataset}old/train/{category}'
    resize_and_save_images(category, folder_path, output_folder_train)

    output_folder_test = f'{our_dataset}old/test/{category}'
    resize_and_save_images(category, folder_path, output_folder_test)
