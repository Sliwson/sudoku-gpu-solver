#include "cpuSolver.h"
#include <algorithm>
#include <iostream>

bool IsPowerOfTwo(u8 x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

void FillMask(u8* sudoku, u8* mask)
{
	for (int i = 0; i < 81; i++)
		if (sudoku[i] > 0)
			mask[i] = 1 << (sudoku[i] - 1);
}

void PropagateHorizontal(u8* mask, int idx, u8 propagationMask)
{
	int left = (idx / 9) * 9;
	int right = left + 9;
	for (int i = left; i < right; i++)
	{
		if (i == idx)
			continue;

		mask[i] &= propagationMask;
	}
}

void PropagateVertical(u8* mask, int idx, u8 propagationMask)
{
	int begin = idx % 9;
	for (int i = begin; i < 81; i += 9)
	{
		if (i == idx)
			continue;

		mask[i] &= propagationMask;
	}
}

void PropagateInSquare(u8* mask, int idx, u8 propagationMatrix)
{
	//centres of the squares
	int x = (idx % 9) / 3 * 3 + 1;
	int y = (idx / 27) * 3 + 1;

	for (int x1 = x - 1; x1 <= x + 1; x1++)
	{
		for (int y1 = y - 1; y1 <= y + 1; y1++)
		{
			int pos = 9 * y1 + x1;
			if (pos == idx)
				continue;

			mask[pos] &= propagationMatrix;
		}
	}
}

bool PropagatePossibilities(u8* mask, bool* propagated)
{
	bool any = false;
	for (int i = 0; i < 81; i++)
	{
		if (!propagated[i] && IsPowerOfTwo(mask[i]))
		{
			any = true;
			propagated[i] = true;

			u8 propagationMask = 0x1ff & ~mask[i];
			PropagateHorizontal(mask, i, propagationMask);
			PropagateVertical(mask, i, propagationMask);
			PropagateInSquare(mask, i, propagationMask);
		}
	}

	return any;
}

void FillResult(u8* mask, u8* result)
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

void SolveCpu(u8 sudoku[81], u8 result[81])
{
	u8 mask[81];
	std::fill_n(mask, 81, 0x1ff);

	bool propagated[81] = { false };

	FillMask(sudoku, mask);

	while (PropagatePossibilities(mask, propagated)) 
	{
		for (int i = 0; i < 81; i++)
			if (mask[i] == 0)
			{
				std::cout << "Can't solve!" << std::endl;
				return;
			}
	}

	//is ambigious
	int amb = 0;
	for (int i = 0; i < 81; i++)
		if (!IsPowerOfTwo(mask[i]))
			amb ++;

	if (amb)
		std::cout << "Have to use backtracking: " << amb << std::endl;

	FillResult(mask, result);
}
