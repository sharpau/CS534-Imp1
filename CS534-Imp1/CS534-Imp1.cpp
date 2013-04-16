// CS534-Imp1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

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

	while(currentError > 0) {
		currentError = batchPerceptron(twoGaussian, weights);
		errors.push_back(currentError);
	}
	random_shuffle(twoGaussian.begin(), twoGaussian.end());






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
	return 1.0f;
}

// takes initial weights, training data, returns weight vector & total SSE error
double votedPerceptron(
	const vector<pair<int, vector<double>>> training, 
	vector<double>& weights
	) {
	return 1.0f;
}