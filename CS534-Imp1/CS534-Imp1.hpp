// function prototypes and includes

const int num_epochs = 10;



void doGradients(
	void
	);

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

void batchPerceptron(
	void
	);

double batchPerceptronEpoch(
	const std::vector<std::pair<int, std::vector<double>>> training, 
	std::vector<double>& weights,
	int & misClass
	);

void votedPerceptron(
	void
	);

int sign(int in) {
	if(in >= 0) { return 1;}
	else { return -1;}
}