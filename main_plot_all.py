from constants import ALL_MODELS
from functions import ModelPlotter, ResultPlotter

if __name__ == "__main__":

    for MODEL in ALL_MODELS:
        # === LOAD MODEL ===
        plotter = ModelPlotter(MODEL)
        plotter.load_model()

        # === IMAGES ===
        plotter.prepare_images()
        plotter.plot()

    # Create comparison plots
    plotter = ResultPlotter(ALL_MODELS)
    plotter.create_all_models_comparison(image_index=0)
    # plotter.create_all_models_comparison(image_index=1)
    # plotter.create_all_models_comparison(image_index=2)