# A
from pandas import read_csv
from keras.models import load_model

actual_data = read_csv("input_data/actual.csv")
nn_representations_data = read_csv("input_data/nn_representations.csv")
print(actual_data)
print(nn_representations_data)

model = load_model('input_data/WindDenseNN.h5')
# summarize the structure of the model
model.summary()
# Get the weights of the first layer of the model
model.layers[0].get_weights()