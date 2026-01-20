from constants import MODEL
from functions import ModelPlotter

if __name__ == "__main__":
    # === LOAD MODEL ===
    plotter = ModelPlotter(MODEL)
    plotter.load_model()

    # === IMAGES ===
    plotter.prepare_images()
    plotter.plot()