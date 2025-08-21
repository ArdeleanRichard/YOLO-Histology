from constants import image_folder, label_folder
from functions import load_model_test, prepare_images, plot, create_all_models_comparison

if __name__ == "__main__":
    for MODEL in ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yoloe", "yolow"]:
        # === LOAD MODEL ===
        model = load_model_test(MODEL)
        class_names = model.names
        num_classes = len(class_names)

        # === IMAGES ===
        combined_images = prepare_images(model, image_folder, label_folder, class_names)
        plot(MODEL, combined_images)

    # Create comparison for first image (index 0)
    create_all_models_comparison(image_folder, label_folder, image_index=0)
    # create_all_models_comparison(image_folder, label_folder, image_index=1)
    # create_all_models_comparison(image_folder, label_folder, image_index=2)