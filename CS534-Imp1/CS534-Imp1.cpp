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
	vector<pair<int, vector<double>>> trainClass;
	ifstream trainClassFile("twogaussian.csv");
	while(trainClassFile.good()) {
		int y;
		double x1, x2;
		trainClassFile >> y;
		trainClassFile.get();
		trainClassFile >> x1;
		trainClassFile.get();
		trainClassFile >> x2;
		trainClassFile.get();

		std::vector<double> example;
		example.push_back(x1);
		example.push_back(x2);

		trainClass.push_back(make_pair(y, example));
	}
	random_shuffle(trainClass.begin(), trainClass.end());

		// read in classification training data
	vector<pair<int, vector<double>>> testClass;
	ifstream testClassFile("iris-twoclass.csv");
	while(testClassFile.good()) {
		int y;
		double x1, x2;
		testClassFile >> y;
		testClassFile.get();
		testClassFile >> x1;
		testClassFile.get();
		testClassFile >> x2;
		testClassFile.get();

		std::vector<double> example;
		example.push_back(x1);
		example.push_back(x2);

		testClass.push_back(make_pair(y, example));
	}
	random_shuffle(testClass.begin(), testClass.end());

	return 0;
}

// takes initial weights, test & training data, returns weight vector & total SSE error
double batchGradientDescent(
	vector<vector<double>> training, 
	vector<vector<double>> test,
	vector<vector<double>>& initWeights, 
	vector<vector<double>>& finalWeights
	) {
	return 1.0f;
}

// takes initial weights, test & training data (already randomized), returns weight vector & total SSE error
double stochasticGradientDescent(
	vector<vector<double>> training, 
	vector<vector<double>> test,
	vector<vector<double>>& initWeights, 
	vector<vector<double>>& finalWeights
	) {
	return 1.0f;
}

// takes initial weights, test & training data, returns weight vector & total SSE error
double batchPerceptron(
	vector<pair<int, vector<double>>> training, 
	vector<pair<int, vector<double>>> test,
	vector<vector<double>>& initWeights, 
	vector<vector<double>>& finalWeights
	) {
	return 1.0f;
}

// takes initial weights, test & training data, returns weight vector & total SSE error
double votedPerceptron(
	vector<pair<int, vector<double>>> training, 
	vector<pair<int, vector<double>>> test,
	vector<vector<double>>& initWeights, 
	vector<vector<double>>& finalWeights
	) {
	return 1.0f;
}