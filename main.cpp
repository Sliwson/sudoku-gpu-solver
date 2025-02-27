#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <array>
#include <chrono>

#include "solver.cuh"
#include "cpuSolver.h"

#define LOG
//#define CPU

using namespace std;

constexpr char* sudokuPath = "zeros.csv";
//constexpr char* sudokuPath = "sudoku.csv";
constexpr int sudokuMaxCout = 0xefffff;

bool IsCorrect(u16* sudoku, u16* answer)
{
	for (int i = 0; i < 81; i++)
		if (answer[i] != sudoku[i])
			return false;

	return true;
}

bool IsCorrect(u16* sudoku)
{
	for (int i = 0; i < 81; i++)
		if (sudoku[i] <= 0 || sudoku[i] > 9)
			return false;

	//horizontal
	array<bool, 9> answers;
	for (int i = 0; i < 9; i++)
	{
		for_each(answers.begin(), answers.end(), [](auto& el) { el = false; });
		for (int j = 0; j < 9; j++)
			answers[sudoku[i * 9 + j] - 1] = true;

		if (std::any_of(answers.begin(), answers.end(), [](const auto& el) { return el == false; }))
			return false;
	}

	//vertical
	for (int i = 0; i < 9; i++)
	{
		for_each(answers.begin(), answers.end(), [](auto& el) { el = false; });
		for (int j = 0; j < 9; j++)
			answers[sudoku[j * 9 + i] - 1] = true;

		if (std::any_of(answers.begin(), answers.end(), [](const auto& el) { return el == false; }))
			return false;
	}

	//squares
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			for_each(answers.begin(), answers.end(), [](auto& el) { el = false; });
			int x1 = 3 * x + 1;
			int y1 = 3 * y + 1;

			for (int xx = x1 - 1; xx <= x1 + 1; xx++)
				for (int yy = y1 - 1; yy <= y1 + 1; yy++)
					answers[sudoku[9 * yy + xx] - 1] = true;

			if (std::any_of(answers.begin(), answers.end(), [](const auto& el) { return el == false; }))
				return false;
		}
	}

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
	
	unsigned long long cpuTime = 0;
	unsigned long long gpuTime = 0;

	while (getline(i, s) && counter <= sudokuMaxCout)
	{
		//read sudoku
		for (int i = 0; i < 81; i++)
			sudoku[i] = s[i] - 48;

		//read solution
		for (int i = 82; i < 163; i++)
			solution[i - 82] = s[i] - 48;

		//GPU
		const auto begin = chrono::high_resolution_clock::now();

		u16 gpuAns[81];
		runKernel(sudoku, gpuAns);

		const auto end = chrono::high_resolution_clock::now();
		const auto duration = chrono::duration_cast<chrono::milliseconds>(end - begin).count();

		//CPU
		const auto beginCpu = chrono::high_resolution_clock::now();

		u16 cpuAns[81];
#ifdef CPU
		SolveCpu(sudoku, cpuAns);
#endif

		const auto endCpu = chrono::high_resolution_clock::now();
		const auto durationCpu = chrono::duration_cast<chrono::milliseconds>(endCpu - beginCpu).count();

		//check result
		bool rGpu = IsCorrect(gpuAns);
		bool rCpu = IsCorrect(cpuAns);
#ifdef LOG
		string resultGpu = rGpu ? "OK" : "WRONG";
		string resultCpu = rCpu ? "OK" : "WRONG";
		cout << "GPU: Sudoku[" << counter << "]: t = " << duration << ", result: " << resultGpu << endl;
#ifdef CPU
		cout << "CPU: Sudoku[" << counter << "]: t = " << durationCpu << ", result: " << resultCpu << endl;
#endif
#endif

		cpuTime += durationCpu;
		gpuTime += duration;

		counter++;
		if (counter >= 10000)
			break;
	}

	cout << "Solved " << counter << " sudoku in gpu = " << gpuTime << "ms" << endl;
	cout << "Gpu preformance: " << ((float)counter / gpuTime) << " sudoku/ms" << endl;

	i.close();
}

int main(int argc, u16** argv)
{
	InitKernel();
	InitCpu();

	SolveFromFile(sudokuPath);

	CleanCpu();
	CleanKernel();

	return 0;
}
