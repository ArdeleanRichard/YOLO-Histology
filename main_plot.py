from constants import MODEL, image_folder, label_folder
from functions import load_model_test, get_file_names, prepare_images, plot

if __name__ == "__main__":
    # === LOAD MODEL ===
    model = load_model_test(MODEL)
    class_names = model.names
    num_classes = len(class_names)

    # === IMAGES ===
    images = get_file_names(image_folder, 4)  # Get 4 images

    combined_images = prepare_images(image_folder, label_folder, class_names)
    plot(combined_images)