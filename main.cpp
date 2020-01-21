#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <chrono>

#include "solver.cuh"
#include "cpuSolver.h"

using namespace std;

constexpr char* sudokuPath = "sudoku.csv";
constexpr int sudokuMaxCout = 0xefffff;

bool IsCorrect(u16* sudoku, u16* answer)
{
	for (int i = 0; i < 81; i++)
		if (answer[i] != sudoku[i])
			return false;

	return true;
}

void SolveFromFile(string filename)
{
	auto i = ifstream(filename);
	if (!i.good())
		return;

	u16 sudoku[81];
	u16 solution[81];

	int counter = 1;
	
	//skip first line
	string s;
	getline(i, s);

	const auto beginAll = chrono::high_resolution_clock::now();
	while (getline(i, s) && counter <= sudokuMaxCout)
	{
		//read sudoku
		for (int i = 0; i < 81; i++)
			sudoku[i] = s[i] - 48;

		//read solution
		for (int i = 82; i < 163; i++)
			solution[i - 82] = s[i] - 48;

		const auto begin = chrono::high_resolution_clock::now();
		//compute

		u16 cpuAns[81];
		//SolveCpu(sudoku, cpuAns);
		runKernel(sudoku, cpuAns);

		const auto end = chrono::high_resolution_clock::now();
		const auto duration = chrono::duration_cast<chrono::milliseconds>(end - begin).count();

		//check result
		bool r = IsCorrect(cpuAns, solution);
		string result = r ? "OK" : "WRONG";
		if (!r)
			cout << "Sudoku[" << counter << "]: t = " << duration << ", result: " << result << endl;

		if (counter % 50000 == 0)
			cout << "Progress: " << counter << "/1000000" << endl;

		counter++;
	}

	const auto endAll = chrono::high_resolution_clock::now();
	const auto duration = chrono::duration_cast<chrono::milliseconds>(endAll - beginAll).count();

	cout << "Solved " << counter << " sudoku in " << duration << " miliseconds" << endl;
	cout << "Preformance: " << counter / duration << " sudoku/ms" << endl;

	i.close();
}

int main(int argc, u16** argv)
{
	SolveFromFile(sudokuPath);
	return 0;
}
