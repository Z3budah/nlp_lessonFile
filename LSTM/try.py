model_path = "model_file_path.h5"
from tensorflow.python.keras.models import load_model
model = load_model(model_path)

print(model.summary())