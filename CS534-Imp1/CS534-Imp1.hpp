double batchPerceptron(
	const std::vector<std::pair<int, std::vector<double>>> training, 
	std::vector<double>& weights
	);

double votedPerceptron(
	const std::vector<std::pair<int, std::vector<double>>> training, 
	std::vector<double>& weights
	);