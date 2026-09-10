"""Shared feature definitions and original experiment defaults."""

TARGET = "Bending strength"
FEATURES = ["Width", "Height", "Density", "Growth-ring width", "Test span", "Dynamic E"]
POSITIVE_FEATURES = ["Density", "Dynamic E"]
NEGATIVE_FEATURES = ["Growth-ring width", "Test span"]
LEARNING_RATE = 0.1
N_ITERATIONS = 120750
PHYSICS_LAMBDA = 0.004
RANDOM_STATE = 42
