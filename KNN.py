# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 00:05:01 2019

@author: dmoro
"""


#from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# Загрузка CSV файла
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader: #все строки из датасета записываем в список
			if not row:#в котором каждый элемент будет содержать строку из датасета
				continue
			dataset.append(row)
	return dataset

# Преобразование строки в вещественные числа
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
        
# разбиение датасета на k частей
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Посчет точности
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Оценка алгоритма с использованием перекрестной проверки
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Вычислене Евклидова расстояния между двумя векторами
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Нахождение k - ближайших соседей
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])#сортировка по расстоянию 
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])#Добавление k - соседей в список
	return neighbors

# Составление прогнозов 
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]#списковое включение
	prediction = max(set(output_values), key=output_values.count)
	return prediction


def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)


#seed(1)
filename = 'iris.data'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


