from pandas import read_csv
import keras
import numpy

# Parse input

nn_representations_data = read_csv("input_data/nn_representations.csv")
model = keras.models.load_model('input_data/WindDenseNN.h5')

# Copy the first layer of the model to a new model

new_model = keras.Sequential()
new_model.add(model.layers[0])

# Keep timestamps

dates = nn_representations_data.values[:,0]

# Remove timestamps from data

nn_representations_data = nn_representations_data.drop(nn_representations_data.columns[0], axis=1)

# Execute the NN

result = new_model.predict(nn_representations_data)

# Write to file

file = open("output_data/new_representation.csv","w")

for i,row in enumerate(result):
	file.write(str(dates[i]))
	file.write(" ")
	for number in row:
		file.write(str(number))
		file.write(" ")
	file.write("\n")
