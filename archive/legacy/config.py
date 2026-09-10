# config.py

SELECTED_FEATURES = [
    "Width",
    "Height",
    "Density",
    "Growth-ring width",
    "Test span",
    "Bending strength",
    "Dynamic E"
]
############################
K_FOLDS = 2
RANDOM_STATE = 42
LEARNING_RATE = 0.1
N_ITERATIONS = 120750


############################

PHYSICS_LAMBDA = 0.004

WIDTH_COL = "Width"
HEIGHT_COL = "Height"
SPAN_COL = "Test span"

############################ Physic Informed LR new it is not working

#STRENGTH_POSITIVITY_LAMBDA = 11
#FORCE_POSITIVITY_LAMBDA = 11

#WIDTH_COL = "Width"
#HEIGHT_COL = "Height"
#SPAN_COL = "Test span"
#TARGET_COL = "Bending strength"

#MIN_BENDING_STRENGTH = 5
#MIN_FORCE = 3