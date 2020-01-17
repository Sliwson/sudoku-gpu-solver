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

bool IsCorrect(u8* sudoku, u8* answer)
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

	u8 sudoku[81];
	u8 solution[81];

	int counter = 1;
	
	//skip first line
	string s;
	getline(i, s);

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

		u8 cpuAns[81];
		SolveCpu(sudoku, cpuAns);

		const auto end = chrono::high_resolution_clock::now();
		const auto duration = chrono::duration_cast<chrono::milliseconds>(end - begin).count();

		//check result
		bool r = IsCorrect(cpuAns, solution);
		string result = r ? "OK" : "WRONG";
		if (!r)
			cout << "Sudoku[" << counter << "]: t = " << duration << ", result: " << result << endl;

		counter++;
	}

	i.close();
}

void runTest(int argc, char **argv)
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    const int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	runKernel();
}

int main(int argc, u8** argv)
{
	SolveFromFile(sudokuPath);
	return 0;
}
