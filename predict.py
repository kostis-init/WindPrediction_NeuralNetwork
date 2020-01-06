from pandas import read_csv
from keras.models import load_model
import numpy

# Parse input

actual_data = read_csv("input_data/actual.csv")
nn_representations_data = read_csv("input_data/nn_representations.csv")
model = load_model('input_data/WindDenseNN.h5')

# Keep timestamps

dates = nn_representations_data.values[:,0]

# Remove timestamps from data

nn_representations_data = nn_representations_data.drop(nn_representations_data.columns[0], axis=1)
actual_data = actual_data.drop(actual_data.columns[0], axis=1)

# Execute the NN

result = model.predict(nn_representations_data)

# Calculate metrics

mae_sum = 0
mse_sum = 0
mape_sum = 0

counter = 0
for i,row in enumerate(result):
	for j,number in enumerate(row):
		diff = numpy.abs(number - actual_data.values[i][j])
		mae_sum += diff
		mse_sum += diff**2
		if actual_data.values[i][j] == 0:
			mape_sum += diff/numpy.average(actual_data.values)
		else:
			mape_sum += diff/actual_data.values[i][j]
		counter += 1


# Write to file

file = open("output_data/predicted.csv","w")

file.write("MAE: ")
file.write(str(mae_sum/counter))
file.write(" MAPE: ")
file.write(str(100 * mape_sum/counter))
file.write(" MSE: ")
file.write(str(mse_sum/counter))
file.write("\n")

for i,row in enumerate(result):
	file.write(str(dates[i]))
	file.write(" ")
	for number in row:
		file.write(str(number))
		file.write(" ")
	file.write("\n")
