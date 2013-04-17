// CS534-Imp1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <assert.h>
#include "CS534-Imp1.hpp"

using namespace std;



int _tmain(int argc, _TCHAR* argv[])
{
	// read in regression training
	vector<vector<double>> trainingData;
	ifstream trainingFile("regression-train.csv");
	while(trainingFile.good()) {
		double x1, x2, x3, y;
		trainingFile >> x1;
		trainingFile.get();
		trainingFile >> x2;
		trainingFile.get();
		trainingFile >> x3;
		trainingFile.get();
		trainingFile >> y;
		trainingFile.get();

		std::vector<double> example;
		example.push_back(x1);
		example.push_back(x2);
		example.push_back(x3);
		example.push_back(y);

		trainingData.push_back(example);
	}
	// randomize order for stochastic
	random_shuffle(trainingData.begin(), trainingData.end());

	// read in regression test
	vector<vector<double>> testData;
	ifstream testFile("regression-test.csv");
	while(testFile.good()) {
		double x1, x2, x3, y;
		testFile >> x1;
		testFile.get();
		testFile >> x2;
		testFile.get();
		testFile >> x3;
		testFile.get();
		testFile >> y;
		testFile.get();

		std::vector<double> example;
		example.push_back(x1);
		example.push_back(x2);
		example.push_back(x3);
		example.push_back(y);

		testData.push_back(example);
	}
	random_shuffle(testData.begin(), testData.end());


	// read in classification training data
	vector<pair<int, vector<double>>> twoGaussian;
	ifstream twoGaussianCSV("twogaussian.csv");
	while(twoGaussianCSV.good()) {
		int y;
		double x1, x2;
		twoGaussianCSV >> y;
		twoGaussianCSV.get();
		twoGaussianCSV >> x1;
		twoGaussianCSV.get();
		twoGaussianCSV >> x2;
		twoGaussianCSV.get();

		std::vector<double> example;
		example.push_back(x1);
		example.push_back(x2);

		twoGaussian.push_back(make_pair(y, example));
	}
	vector<double> errors;
	double currentError;

	// initial weight 0 for 2 features
	vector<double> weights(2, 0);

	do {
		currentError = batchPerceptron(twoGaussian, weights);
		errors.push_back(currentError);
	} while(abs(currentError) > 0.07); // never converges to <.06, but <.07 works
	// TODO: write weights, currentError to CSV files

	// read in classification training data
	vector<pair<int, vector<double>>> irisData;
	ifstream irisCSV("iris-twoclass.csv");
	while(irisCSV.good()) {
		int y;
		double x1, x2;
		irisCSV >> y;
		irisCSV.get();
		irisCSV >> x1;
		irisCSV.get();
		irisCSV >> x2;
		irisCSV.get();

		std::vector<double> example;
		example.push_back(x1);
		example.push_back(x2);

		irisData.push_back(make_pair(y, example));
	}
	random_shuffle(irisData.begin(), irisData.end());

	return 0;
}

// takes initial weights, test & training data, returns weight vector & total SSE error
double batchGradientDescent(
	vector<vector<double>> training, 
	vector<vector<double>> test,
	vector<double>& initWeights, 
	vector<double>& finalWeights
	) {
	return 1.0f;
}

// takes initial weights, test & training data (already randomized), returns weight vector & total SSE error
double stochasticGradientDescent(
	vector<vector<double>> training, 
	vector<vector<double>> test,
	vector<double>& initWeights, 
	vector<double>& finalWeights
	) {
	return 1.0f;
}

// takes initial weights, training data, returns weight vector & total SSE error
double batchPerceptron(
	const vector<pair<int, vector<double>>> training, 
	vector<double>& weights
	) {
	// delta vector as 0s
	vector<double> delta(weights.size(), 0);

	// go through all examples
	for(auto example : training) {
		// make sure none of the examples are invalid
		assert(weights.size() == example.second.size());

		double um = 0;
		for(int i = 0; i < weights.size(); i++) {
			um += (weights[i] * example.second[i]);
		}
		if((example.first * um) <= 0) {
			for(int i = 0; i < delta.size(); i++) {
				delta[i] -= example.first * example.second[i];
			}
		}
	}

	double norm = 0;
	for(int i = 0; i < delta.size(); i++) {
		delta[i] /= training.size();
		weights[i] -= delta[i]; // learning rate = 1
		norm += delta[i] * delta[i]; // sum of squares for vector norm
	}
	norm = sqrt(norm); // sqrt of sum of squares for vector norm
	

	return norm;
}

// takes initial weights, training data, returns weight vector & total SSE error
double votedPerceptron(
	const vector<pair<int, vector<double>>> training, 
	vector<double>& weights
	) {
	return 1.0f;
}