from constants import MODEL, image_folder, label_folder
from functions import load_model_test, prepare_images, plot

if __name__ == "__main__":
    # === LOAD MODEL ===
    model = load_model_test(MODEL)
    class_names = model.names
    num_classes = len(class_names)

    # === IMAGES ===
    combined_images = prepare_images(model, image_folder, label_folder, class_names)
    plot(MODEL, combined_images)