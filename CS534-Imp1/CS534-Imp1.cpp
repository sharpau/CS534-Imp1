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
	random_shuffle(trainingData.begin(), trainingData.end());

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
	return 0;
}

