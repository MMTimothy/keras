import os
from library import file_finder

KALES = os.path.join("dataset","kales")
kales = file_finder.get_list_files(KALES)
SPINACH = os.path.join("dataset","spinach")
spinach = file_finder.get_list_files(SPINACH)

CLASSES = ["KALES","SPINACH"]

TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

INIT_LR = 1e-9
BATCH_SIZE = 64
NUM_EPOCHS = 50

MODEL_PATH = os.path.sep.join(["output","kales_spinach.model"])
TRAINING_PLOT_PATH = os.path.sep.join(["output","training_plot.png"])

OUTPUT_IMAGE_PATH = os.path.sep.join(["output","examples"])

