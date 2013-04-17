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
	vector<double> batchTrainingError;
	double batchTestError;

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
	vector<double> &weights,
	double learningRate
	) {
		int i, j;
		double result1, result2, sse;
		vector<double> gradient (weights.size());
		vector<double>::iterator wIterate;
		
		//  psuedocode:
		// [outside: init weights to 0]
		//	sse = 0
		//	foreach (weights(size)) gradient = 0
		//	run foreach(i) training: 
		//			result1 = 0;
		//		for (j) from 0 to weights(size-1) 
		//			result1 += weights[j] * training[i][j]
		//			result2 = result1 - training[i][weights(size-1)]
		//		sse += 0.5 * pow(result2, 2)
		//		for (j) from 0 to training[i] size-2
		//			gradient[j] += result2 * training[i][j]
		//	foreach(i) weights
		//		weights[i] = weights[i] - (learnrate) * gradient[i]
		//	[return from function, then store sse in vector.  End program by outputting error vector.]
		sse = 0;
		for(i = 0; i<weights.size(); i++){
			gradient[i] = 0;
		}
		for(i = 0; i<training.size(); i++){
			result1 = 0;
			for(j = 0; j<weights.size(); j++){
				result1 += weights[j] * training[i][j];
				result2 = result1 - training[i][weights.size()-1];
			}
			sse += 0.5 * pow(result2, 2);
			for(j = 0; j<training.size()-1; j++){
				gradient[j] += result2 * training[i][j];
			}
		}
		for(i = 0; i<weights.size(); i++){
			weights[i] = weights[i] - learningRate * gradient[i];
		}

	return sse;
}

// takes initial weights, test & training data (already randomized), returns weight vector & total SSE error
double stochasticGradientDescent(
	vector<vector<double>> training, 
	vector<double> &weights,
	double learningRate,
	) {

		int i, j;
		double result1, result2, sse;
		vector<double> gradient (weights.size());
		vector<double>::iterator wIterate;
		
		//  psuedocode:
		// [outside: init weights to 0]
		//	sse = 0
		//	foreach (weights(size)) gradient = 0
		//	run foreach(i) training: 
		//			result1 = 0;
		//		for (j) from 0 to weights(size-1) 
		//			result1 += weights[j] * training[i][j]
		//			result2 = result1 - training[i][weights(size-1)]
		//		sse += 0.5 * pow(result2, 2)
		//		for (j) from 0 to training[i] size-2
		//			gradient[j] += result2 * training[i][j]
		//		foreach(j) weights
		//			weights[j] = weights[j] - (learnrate) * gradient[j]
		//		
		//	[return from function, then store sse in vector.  End program by outputting error vector.]
		sse = 0;
		for(i = 0; i<weights.size(); i++){
			gradient[i] = 0;
		}
		for(i = 0; i<training.size(); i++){
			result1 = 0;
			for(j = 0; j<weights.size(); j++){
				result1 += weights[j] * training[i][j];
				result2 = result1 - training[i][weights.size()-1];
			}
			sse += 0.5 * pow(result2, 2);
			for(j = 0; j<training.size()-1; j++){
				gradient[j] += result2 * training[i][j];
			}
			for(j = 0; j<weights.size(); j++){
				weights[j] = weights[j] - learningRate * gradient[j];
			}
		}

	// note: for simplicity, this is the sse per epoch.  
	// more formally, an sse could be produced per update step
	// we chose this method for consistency of comparison with the batch algorithm
	return sse;
}

double testGradientDescent(
		vector<vector<double>> test,
		vector<double> &weights
	){
		int i, j;
		double sse, result1, result2;

		sse = 0;
		for(i = 0; i<test.size(); i++){
			result1 = 0;

			for(j = 0; j<weights.size(); j++){
				result1 += weights[j] * test[i][j];
				result2 = result1 - test[i][weights.size()-1];
			}
			sse += 0.5 * pow(result2, 2);
		}

		return sse;
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