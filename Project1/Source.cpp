#include <iostream>
#include <Eigen/Dense> 
#include <vector>

using namespace Eigen; 

MatrixXd createWeights(int, int);
MatrixXd train(MatrixXd, std::vector<int>, int);

int main() {
	MatrixXd weights = { createWeights(5,5) };
	MatrixXd new_weights = { train(weights, { -1, 1, 1, -1, -1}, 5) };
	MatrixXd d_new_weights = { train(new_weights,{ -1, -1, -1, -1, -1 } , 5)};
	std::cout << d_new_weights << std::endl;
	return 0;
}

MatrixXd createWeights(int i, int j) {
	MatrixXd weights = { MatrixXd::Zero(i, j) };
	return weights; 
}

MatrixXd train(MatrixXd oldWeights, std::vector<int> pattern, int i) {
	VectorXd v_pattern(i);
	for (auto x = 0; x < i; x++) { 
		v_pattern(x) = pattern[x];
	}
	MatrixXd newWeights = v_pattern * v_pattern.transpose() - MatrixXd::Identity(i,i);
	return oldWeights + newWeights;
}