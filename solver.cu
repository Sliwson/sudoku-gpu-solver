#include "solver.cuh"
#include <stdio.h>

namespace {
	void FillMask(u16* sudoku, u16* mask)
	{
		for (int i = 0; i < 81; i++)
			if (sudoku[i] > 0)
				mask[i] = 1 << (sudoku[i] - 1);
			else
				mask[i] = 0x1ff;
	}

	void FillResult(u16* mask, u16* result)
	{
		for (int i = 0; i < 81; i++)
		{
			int m = 1;
			for (int r = 1; r <= 9; r++)
			{
				if (m << (r - 1) == mask[i])
					result[i] = r;
			}
		}
	}
	
	u16* d_sudoku;
	bool* d_propagated;
	int* d_helperInt;
}

__device__ bool d_kernelBool;

__device__ void Clamp(int &val, const int &min, const int &max)
{
	if (val < min)
		val = min;
	if (val > max);
		val = max;
}

__device__ bool IsPowerOfTwo(const u16 &x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

__device__ int CountOnes(const u16& x)
{
	int count = 0;
	for (int i = 0; i <= 9; i++)
	{
		if (((x >> i) & 1) == 1)
			count++;
	}

	return count;
}

__global__ void Propagate(u16* d_mask, bool* d_propagated)
{
	d_kernelBool = false;

    const int tid = threadIdx.x;
	const int maskIdx = blockIdx.x;

	//set up startup point on matrix
	d_mask += maskIdx;

	if (tid >= 81)
		return;

	__shared__ u16 s[81];
	s[tid] = d_mask[tid];
	__syncthreads();

	if (!d_propagated[tid] && IsPowerOfTwo(s[tid]))
	{
		d_kernelBool = true;
		d_propagated[tid] = true;
		u16 propagationMask = 0x1ff & ~s[tid];

		//vertical
		int pos = (tid + 9) % 81;

		for (int i = 0; i < 8; i++)
		{
			s[pos] &= propagationMask;
			pos = (pos + 9) % 81;
		}

		//horizontal
		int left = (tid / 9) * 9;
		int right = left + 8;

		pos = tid + 1;
		if (pos > right)
			pos = left;

		for (int i = 0; i < 8; i++)
		{
			
			s[pos] &= propagationMask;
			pos++;
			if (pos > right)
				pos = left;
		}
		
		//in square
		int sx = (tid % 9) / 3 * 3 + 1;
		int sy = (tid / 27) * 3 + 1;
		
		int x = tid % 9;
		int y = tid / 9;
		for (int i = 0; i < 8; i++)
		{
			x++;
			if (x > sx + 1)
			{
				x = sx - 1;
				y++;
				if (y > sy + 1)
					y = sy - 1;
			}

			int p = 9 * y + x;
			s[p] &= propagationMask;
		}
	}

	__syncthreads();
	d_mask[tid] = s[tid];
}

__global__ void FindLowest(u16* d_mask, int* helperInt)
{
	int tid = threadIdx.x;
	int sudokuIdx = blockIdx.x;
	d_mask += sudokuIdx;

	if (tid >= 81)
		return;

	__shared__ u16 s[81 + 64];
	s[tid] = CountOnes(d_mask[tid]);

	if (s[tid] == 1)
		s[tid] = 10;

	if (tid < 64)
		s[tid + 81] = tid;

	int working = 64;
	while (working > 1)
	{
		if (tid < working)
		{
			int targetIdx = working * 2 - tid - 1;
			if (targetIdx < 81 && s[targetIdx] < s[tid])
			{
				s[tid] = s[targetIdx];
				s[tid + 81] = targetIdx;
			}
		}

		working >>= 1;
		__syncthreads();
	}

	if (tid == 0)
	{
		helperInt[2 * sudokuIdx] = s[0];
		helperInt[2 * sudokuIdx + 1] = s[81];
	}
}

void InitKernel()
{
	cudaMalloc(&d_propagated, 81 * 1000 * sizeof(bool));
	cudaMalloc(&d_sudoku, 81 * 1000 * sizeof(u16));
	cudaMalloc(&d_helperInt, 2000 * sizeof(int));
}

void CleanKernel()
{
	cudaFree(d_sudoku);
	cudaFree(d_propagated);
	cudaFree(d_helperInt);
}

void runKernel(u16 sudoku[81], u16 result[81])
{
	u16 mask[81];
	FillMask(sudoku, mask);
	int activeMasks = 1;

	cudaMemcpy(d_sudoku, mask, 81 * sizeof(u16), cudaMemcpyHostToDevice);
	cudaMemset(d_propagated, false, 81 * 1000 * sizeof(bool));
	cudaMemset(d_helperInt, 0, 2000 * sizeof(int));

	while (true)
	{
		int activeMasksNew = activeMasks;

		//propagate all sudokus
		while (true)
		{
			Propagate <<<activeMasks, 128>> > (d_sudoku, d_propagated);
			bool anyChanged;
			cudaMemcpyFromSymbol(&anyChanged, d_kernelBool, sizeof(anyChanged), 0, cudaMemcpyDeviceToHost);
			if (!anyChanged)
				break;
		}

		//check for split
		FindLowest << <activeMasks, 128 >> > (d_sudoku, d_helperInt);
		int testSplit[2];
		cudaMemcpy(testSplit, d_helperInt, activeMasks * 2 * sizeof(int), cudaMemcpyDeviceToHost);

		if (testSplit[0] == 10)
			break;
	}
	
	cudaMemcpy(mask, d_sudoku, 81 * sizeof(u16), cudaMemcpyDeviceToHost);
	FillResult(mask, result);
}
