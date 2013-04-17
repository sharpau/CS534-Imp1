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