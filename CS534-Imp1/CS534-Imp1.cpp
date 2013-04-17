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

const int num_epochs = 10;

using namespace std;

// function prototypes
double batchGradientDescent(
	vector<vector<double>> training, 
	vector<double> &weights,
	double learningRate
	);
double stochasticGradientDescent(
	vector<vector<double>> training, 
	vector<double> &weights,
	double learningRate
	);
double testGradientDescent(
		vector<vector<double>> test,
		vector<double> &weights
	);

int _tmain(int argc, _TCHAR* argv[])
{
	// loop var
	int i;

	// read in regression training
	double gradientLearningRate;
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

	vector<double> batchGradientWeights (trainingData[0].size()-1);
	vector<double> stochasticGradientWeights (trainingData[0].size()-1);
	vector<double> batchGradientTrainingError (num_epochs);
	vector<double> stochasticGradientTrainingError (num_epochs);

	//init weight vectors
	for(vector<double>::size_type i = 0; i<trainingData[0].size()-1; i++){
		batchGradientWeights[i] = 0;
		stochasticGradientWeights[i] = 0;
	}

	// run batch gradient descent training
	for(i = 0; i<num_epochs; i++){
		gradientLearningRate = (1 / trainingData.size());
		batchGradientTrainingError.push_back(batchGradientDescent(trainingData, batchGradientWeights, gradientLearningRate));
	}
	// run stochastic gradient descent training
	for(i = 0; i<num_epochs; i++){
		// randomize order for stochastic
		random_shuffle(trainingData.begin(), trainingData.end());
		stochasticGradientTrainingError.push_back(stochasticGradientDescent(trainingData, stochasticGradientWeights, gradientLearningRate));
	}

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
	
	double batchGradientTestError;
	double stochasticGradientTestError;

	// run accuracy test trials
	batchGradientTestError = testGradientDescent(testData, batchGradientWeights);
	stochasticGradientTestError = testGradientDescent(testData, stochasticGradientWeights);

	// DO THIS: output trial results in some format

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
	vector<int> misClassHistory;
	int misClass;

	// initial weight 0 for 2 features
	vector<double> weights(2, 0);

	do {
		currentError = batchPerceptron(twoGaussian, weights, misClass);
		errors.push_back(currentError);
		misClassHistory.push_back(misClass);
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

	vector<double> vWeights(2, 0);
	vector<double> errorHistory;
	vector<vector<double>> weightHistory;
	weightHistory.push_back(vWeights);
	vector<int> c;
	c.push_back(0);
	int n = 0;
	for(int t = 0; t < 100; t++) {
		// randomize for each epoch
		random_shuffle(irisData.begin(), irisData.end());

		for(auto example : irisData) {
			double um = 0;
			for(int i = 0; i < vWeights.size(); i++) {
				um += (vWeights[i] * example.second[i]);
			}
			if((um * example.first) <= 0) {
				for(int i = 0; i < vWeights.size(); i++) {
					vWeights[i] += (example.second[i] * example.first);
				}
				n++;
				c[t] = 0;
			}
			else {
				c[t]++;
			}
		}
		weightHistory.push_back(vWeights);
		// time to vote!
		errorHistory.push_back(0);
		for(auto example : irisData) {
			double voteResult = 0;
			for(int i = 0; i < t + 2; i++) {
				double thisVote = 0;
				for(int j = 0; j < vWeights.size(); j++) {
					thisVote += weightHistory[t][j] * example.second[j];
				}
				voteResult += c[i] * sign(thisVote);
			}
			if(example.first != sign(voteResult)) {
				errorHistory[t]++;
			}
		}
	}

	return 0;
}

// takes initial weights, test & training data, returns weight vector & total SSE error
double batchGradientDescent(
	vector<vector<double>> training, 
	vector<double> &weights,
	double learningRate
	) {
		double result1, result2, sse;
		vector<double> gradient (weights.size());
		
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
		for(vector<double>::size_type i = 0; i<weights.size(); i++){
			gradient[i] = 0;
		}
		for(vector<vector<double>>::size_type i = 0; i<training.size(); i++){
			result1 = 0;
			for(vector<double>::size_type j = 0; j<weights.size(); j++){
				result1 += weights[j] * training[i][j];
				result2 = result1 - training[i][weights.size()-1];
			}
			sse += 0.5 * pow(result2, 2);
			for(vector<double>::size_type j = 0; j<weights.size(); j++){
				gradient[j] += result2 * training[i][j];
			}
		}
		for(vector<double>::size_type i = 0; i<weights.size(); i++){
			weights[i] = weights[i] - learningRate * gradient[i];
		}

	return sse;
}

// takes initial weights, test & training data (already randomized), returns weight vector & total SSE error
double stochasticGradientDescent(
	vector<vector<double>> training, 
	vector<double> &weights,
	double learningRate
	) {
		double result1, result2, sse;
		vector<double> gradient (weights.size());
		
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
		for(vector<double>::size_type i = 0; i<weights.size(); i++){
			gradient[i] = 0;
		}
		for(vector<vector<double>>::size_type i = 0; i<training.size(); i++){
			result1 = 0;
			for(vector<double>::size_type j = 0; j<weights.size(); j++){
				result1 += weights[j] * training[i][j];
				result2 = result1 - training[i][weights.size()-1];
			}
			sse += 0.5 * pow(result2, 2);
			for(vector<double>::size_type j = 0; j<weights.size(); j++){
				gradient[j] += result2 * training[i][j];
			}
			for(vector<double>::size_type j = 0; j<weights.size(); j++){
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
		double sse, result1, result2;

		sse = 0;
		for(vector<vector<double>>::size_type i = 0; i<test.size(); i++){
			result1 = 0;

			for(vector<double>::size_type j = 0; j<weights.size(); j++){
				result1 += weights[j] * test[i][j];
				result2 = result1 - test[i][weights.size()-1];
			}
			sse += 0.5 * pow(result2, 2);
		}

		return sse;
}

// takes initial weights, training data, returns weight vector & error (norm of delta)
// runs through data set once
double batchPerceptron(
	const vector<pair<int, vector<double>>> training, 
	vector<double>& weights,
	int & misClass
	) {
	// delta vector as 0s
	vector<double> delta(weights.size(), 0);
	misClass = 0;

	// go through all examples
	for(auto example : training) {
		// make sure none of the examples are invalid
		assert(weights.size() == example.second.size());

		double um = 0;
		for(int i = 0; i < weights.size(); i++) {
			um += (weights[i] * example.second[i]);
		}
		if((example.first * um) <= 0) {
			misClass++;
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