#include <iostream>
#include <Eigen/Dense> 
#include <vector>

using namespace Eigen; 

MatrixXd createWeights(int, int);
MatrixXd train(MatrixXd, std::vector<int>, int);
int getEnergy(MatrixXd, int, std::vector<int>);
VectorXd vec_to_vec(std::vector<int>, int);
int sign(int);

int main() {
	MatrixXd weights { createWeights(5,5) };
	MatrixXd new_weights { train(weights, { -1, 1, 1, -1, 1}, 5) };
	MatrixXd d_new_weights { train(new_weights,{ 1, -1, 1, -1, 1 }, 5) };
	std::cout << d_new_weights << std::endl;
	int energy { getEnergy(d_new_weights, 3, { 1, 1, 1, 1, 1}) };
	std::cout << "Energy of third node is " << energy << std::endl;
	return 0;
}

MatrixXd createWeights(int i, int j) {
	MatrixXd weights { MatrixXd::Zero(i, j) };
	return weights; 
}

MatrixXd train(MatrixXd oldWeights, std::vector<int> pattern, int i) {
	VectorXd v_pattern { vec_to_vec(pattern, i) };
	MatrixXd newWeights { v_pattern * v_pattern.transpose() - MatrixXd::Identity(i,i) };
	return oldWeights + newWeights;
}

int getEnergy(MatrixXd weights, int i, std::vector<int> pattern) {
	VectorXd v_pattern { vec_to_vec(pattern, weights.col(0).size()) };
	int energy = weights.col(i - 1).dot(v_pattern.transpose()) ;
	return energy;
}

int sign(int energy) {
	if (energy >= 0)
		return 1;
	else return -1;
}

VectorXd vec_to_vec(std::vector<int> pattern, int i) {
	VectorXd v_pattern(i);
	for (auto x = 0; x < i; x++) {
		v_pattern(x) = pattern[x];
	}
	return v_pattern;
}