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
}

__device__ bool d_anyChanged;

__global__ void testKernel()
{
    const unsigned int tid = threadIdx.x;
    const unsigned int num_threads = blockDim.x;
}

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

__global__ void Propagate(u16* d_mask, bool* d_propagated, int maskIdx)
{
    const int tid = threadIdx.x;
	if (tid >= 81)
		return;

	d_anyChanged = false;
	if (d_propagated[tid] || !IsPowerOfTwo(d_mask[tid]))
		return;

	d_anyChanged = true;
	d_propagated[tid] = true;
	u16 propagationMask = 0x1ff & ~d_mask[tid];

	//vertical
	int pos = (tid + 9) % 81;

	for (int i = 0; i < 8; i++)
	{
		d_mask[pos] &= propagationMask;
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
		
		d_mask[pos] &= propagationMask;
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
		d_mask[p] &= propagationMask;
	}
}

void InitKernel()
{
	cudaMalloc(&d_propagated, 81 * sizeof(bool));
	cudaMalloc(&d_sudoku, 81 * sizeof(u16));
}

void CleanKernel()
{
	cudaFree(d_sudoku);
	cudaFree(d_propagated);
}

void runKernel(u16 sudoku[81], u16 result[81])
{
	u16 mask[81];
	FillMask(sudoku, mask);

	cudaMemcpy(d_sudoku, mask, 81 * sizeof(u16), cudaMemcpyHostToDevice);
	cudaMemset(d_propagated, false, 81 * sizeof(bool));

	// setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(128, 1, 1);

    // execute the kernel
	while (true)
	{
		Propagate << <grid, threads >> > (d_sudoku, d_propagated, 0);
		bool anyChanged;
		cudaMemcpyFromSymbol(&anyChanged, d_anyChanged, sizeof(anyChanged), 0, cudaMemcpyDeviceToHost);
		if (!anyChanged)
			break;
	}

	cudaMemcpy(mask, d_sudoku, 81 * sizeof(u16), cudaMemcpyDeviceToHost);


	FillResult(mask, result);
}
