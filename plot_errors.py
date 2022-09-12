import pickle
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

with open("epochs.pickle", "rb") as target:
    epochs = pickle.load(target)

with open("train_error.pickle", "rb") as target:
    train_error = pickle.load(target)

with open("dev_error.pickle", "rb") as target:
    dev_error = pickle.load(target)

plt.plot(epochs, train_error, label="Training Data")
plt.plot(epochs, dev_error, label="Validation Data")
plt.title("Model Error Over Time\nWith Dropout=0.5")
plt.xlabel("Epoch")
plt.ylabel("Root Mean Square Error")
plt.xlim(epochs[0], epochs[-1])
plt.legend()
plt.grid(b=True, which="major", color="grey")
plt.minorticks_on()

plt.grid(b=True, which="minor", color="lightgrey")
plt.show()
