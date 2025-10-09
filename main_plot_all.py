from functions import ModelPlotter, ResultPlotter

if __name__ == "__main__":
    all_models = ["rtdetr", "yolo8", "yolo9", "yolo10", "yolo11", "yolo12", "yoloe", "yolow"]

    for MODEL in all_models:
        # === LOAD MODEL ===
        plotter = ModelPlotter(MODEL)
        plotter.load_model()

        # === IMAGES ===
        plotter.prepare_images()
        plotter.plot()

    # Create comparison plots
    plotter = ResultPlotter(all_models)
    plotter.create_all_models_comparison(image_index=0)
    # plotter.create_all_models_comparison(image_index=1)
    # plotter.create_all_models_comparison(image_index=2)