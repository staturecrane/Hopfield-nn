#pragma once
#include <iostream>
#include <Eigen/Dense> 
#include <vector>
#include <random>

using namespace Eigen;

MatrixXd createWeights(int, int);
MatrixXd train(MatrixXd, std::vector<int>, int);
double getEnergy(MatrixXd, int, VectorXd);
VectorXd vec_to_vec(std::vector<int>, int);
double sign(double);
VectorXd recover(MatrixXd, VectorXd);

MatrixXd createWeights(int i, int j) {
	MatrixXd weights{ MatrixXd::Zero(i, j) };
	return weights;
}

MatrixXd train(MatrixXd oldWeights, std::vector<int> pattern, int i) {
	VectorXd v_pattern{ vec_to_vec(pattern, i) };
	MatrixXd newWeights{ v_pattern * v_pattern.transpose() - MatrixXd::Identity(i,i) };
	return oldWeights + newWeights;
}

double getEnergy(MatrixXd weights, int i, VectorXd v_pattern) {
	double energy{ weights.col(i).dot(v_pattern.transpose()) };
	return energy;
}

double sign(double energy) {
	if (energy >= 0)
		return 1.0;
	else return -1.0;
}

VectorXd vec_to_vec(std::vector<int> pattern, int i) {
	VectorXd v_pattern(i);
	for (int x = 0; x < i; x++) {
		v_pattern(x) = pattern[x];
	}
	return v_pattern;
}

VectorXd recover(MatrixXd weights, std::vector<int> pattern) {
	int unchanged{ 0 };
	VectorXd v_pattern {vec_to_vec(pattern, pattern.size()) };
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> u(0, v_pattern.size() - 1);

	for (auto x = 0; x < (v_pattern.size() * 4) || unchanged >= 5; x++) {
		int index{ u(rng) };
		double activation{ sign(getEnergy(weights, index, v_pattern)) };
		if (activation == 0)
			unchanged += 1; 
		else unchanged = 0; 
		v_pattern(index) =  activation;
	}
	return v_pattern;
}