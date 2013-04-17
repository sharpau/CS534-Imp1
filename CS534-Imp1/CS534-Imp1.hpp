double batchPerceptron(
	const std::vector<std::pair<int, std::vector<double>>> training, 
	std::vector<double>& weights,
	int & misClass
	);

double votedPerceptron(
	const std::vector<std::pair<int, std::vector<double>>> training, 
	std::vector<double>& weights
	);

int sign(int in) {
	if(in >= 0) { return 1;}
	else { return -1;}
}