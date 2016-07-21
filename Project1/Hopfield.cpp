#include "Hopfield.h"

int main() {
	MatrixXd weights { createWeights(5,5) };
	MatrixXd new_weights { train(weights, { -1, 1, 1, -1, 1}, 5) };
	MatrixXd d_new_weights { train(new_weights,{ 1, -1, 1, -1, 1 }, 5) };
	std::cout << d_new_weights << std::endl;
	VectorXd recovered = recover(d_new_weights, { 1, 1, 1, 1, 1 });
	std::cout << "Recovered pattern is " << recovered.transpose() << std::endl;
	return 0;
}