#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <chrono>

#include "solver.cuh";

using namespace std;

constexpr char* sudokuPath = "sudoku.csv";
constexpr int sudokuMaxCout = 100;

bool IsCorrect(char* sudoku, char* answer)
{
	for (int i = 0; i < 81; i++)
		if (answer[i] != sudoku[i])
			return false;

	return true;
}

bool SolveFromFile(string filename)
{
	auto i = ifstream(filename);
	if (!i.good())
		return false;

	char sudoku[81];
	char solution[81];

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
		//...

		const auto end = chrono::high_resolution_clock::now();
		const auto duration = chrono::duration_cast<chrono::milliseconds>(end - begin).count();

		//check result
		string result = IsCorrect(sudoku, solution) ? "OK" : "WRONG";
		cout << "Sudoku[" << counter << "]: t = " << duration << ", result: " << result << endl;

		counter++;
	}
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

int main(int argc, char** argv)
{
	SolveFromFile(sudokuPath);
}
