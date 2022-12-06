import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

my_model = models.load_model("my_model2.h5")

output=my_model.predict([np.zeros((1,100,70,3))])

print(output)
