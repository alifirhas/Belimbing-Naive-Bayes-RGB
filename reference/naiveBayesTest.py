from csv import reader
from math import sqrt
from math import exp
from math import pi
import pandas
import numpy as np

# Load a CSV file
def load_csv(filename: str):
	"""Mengambil data pada .csv

	Args:
		filename (string): directory file

	Returns:
		list: semua data dalam bentuk list
	"""
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset: list, column: int):
	for row in dataset:
		row[column] = float(row[column].strip())

# Persiapkan data
def get_data(dataset: list):
    newDataset = dataset.copy()
    newDataset.pop(0)          # Hapus column field

    # ubah tipe data list ke float, kecuali label
    for i in range(len(newDataset[0])-1):		# -1 karena label berada di paling akhir dan tidak perlu dikoversi
        str_column_to_float(newDataset, i)		# Proses konversi tipe data
    
    for row in newDataset:                      # Hapus id
        row.pop(0)
    
    return newDataset

# Ambil nama field yang ada dan ubah ke integer
def get_field(dataset: list):
    # field = []
    # for i in range(len(dataset.pop(0))):
    #     field.append(i)
    
    newDataset = dataset.copy() # salin data
    field = newDataset.pop(0)   # Ambil column pertama
    field.pop(0)                # Tidak perlu id
    return field

# Ambil label yang ada pada data
def get_label(dataset: list):
    newDataset = dataset.copy()
    newDataset.pop(0)
    
    field_list = []
    for row in newDataset:
        field_list.append(row[-1])

    return list(set(field_list))

def field_to_int(field: list):
    fieldInt = []
    for i, field in enumerate(field):
        fieldInt.append(i)
    
    return fieldInt

# Hitung mean di row menggunakan numpy
def calculate_col_mean(a: list):
    a = np.array(a)
    average_col = a.mean(axis=0)
    average_col = [round(i,2) for i in average_col]
    return average_col

# Hitung standard deviation di row menggunakan numpy
# def calculate_col_stdv(a: list):
#     a = np.array(a)
#     stdv_col = a.std(axis=0)
#     stdv_col = [round(i,2) for i in stdv_col]
#     return stdv_col

def calculate_col_stdv(y: list, mean_list: list):
    devian = []
    for j, col in enumerate(y):
        if j >= len(mean_list):
            break
        m = [i[j] for i in y]
        dev = [i-mean_list[j] for i in m]
        dev = [pow(i,2) for i in dev]
        devian.append(dev)
    a = np.array(devian)
    result = list(np.mean(a, axis=1))

    return [round(i,2) for i in result]

# Make a prediction with Naive Bayes on Iris Dataset
filename = '/home/alifirhas/0_Work/0_Kuliah/Semester 5/0_Project/0_Belimbing/Belimbing-Naive-Bayes-RGB/data/Iris.csv'
dataset = load_csv(filename)        # Load data csv

data = get_data(dataset)            # Persiapakan data

field = get_field(dataset)          # Ambil field yang ada
print("Field: ", field_to_int)

label = get_label(dataset)          # Ambil label yang ada
print("Label: ", label)

# Hitung mean setiap label->field

mean_list = []
stdv_list = []
for label in label:
    k = []
    for i, row in enumerate(data):
        if row[4] == label:
            k.append(row[:-1])
    mean = calculate_col_mean(k)
    stdv = calculate_col_stdv(k, mean)
    mean_list.append(mean)
    stdv_list.append(stdv)
    # print(label, ' mean: ', mean)
    # print(label, ' stdv: ', stdv)

print(mean_list)
print()
print(stdv_list)