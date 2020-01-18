#include "cpuSolver.h"
#include <algorithm>
#include <iostream>

bool IsPowerOfTwo(u16 x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

int CountOnes(u16 x)
{
	int count = 0;
	for (int i = 0; i <= 9; i++)
	{
		if (((x >> i) & 1) == 1)
			count++;
	}

	return count;
}

void FillMask(u16* sudoku, u16* mask)
{
	for (int i = 0; i < 81; i++)
		if (sudoku[i] > 0)
			mask[i] = 1 << (sudoku[i] - 1);
}

int FindIdexWithLowest(u16* mask, int& count, int maskIdx)
{
	mask += maskIdx * 81;

	int minIdx = 0;
	int minValue = CountOnes(mask[0]);
	for (int i = 1; i < 81; i++)
	{
		int value = CountOnes(mask[i]);
		if (value < minValue)
		{
			minValue = value;
			minIdx = i;
		}
	}

	count = minValue;
	return minIdx;
}

int FindIdexWithLowestGreaterThanOne(u16* mask, int& count, int maskIdx)
{
	mask += maskIdx * 81;

	int minIdx = -1;
	int minValue = 10;
	for (int i = 0; i < 81; i++)
	{
		int value = CountOnes(mask[i]);
		if (value < minValue && value > 1)
		{
			minValue = value;
			minIdx = i;
		}
	}

	count = minValue;
	return minIdx;
}



void PropagateHorizontal(u16* mask, int idx, u16 propagationMask)
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

void PropagateVertical(u16* mask, int idx, u16 propagationMask)
{
	int begin = idx % 9;
	for (int i = begin; i < 81; i += 9)
	{
		if (i == idx)
			continue;

		mask[i] &= propagationMask;
	}
}

void PropagateInSquare(u16* mask, int idx, u16 propagationMatrix)
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

bool PropagatePossibilities(u16* mask, bool* propagated, int maskIdx)
{
	bool any = false;

	mask += 81 * maskIdx;
	propagated += 81 * maskIdx;

	for (int i = 0; i < 81; i++)
	{
		if (!propagated[i] && IsPowerOfTwo(mask[i]))
		{
			any = true;
			propagated[i] = true;

			u16 propagationMask = 0x1ff & ~mask[i];
			PropagateHorizontal(mask, i, propagationMask);
			PropagateVertical(mask, i, propagationMask);
			PropagateInSquare(mask, i, propagationMask);
		}
	}

	return any;
}

void FillResult(u16* mask, u16* result, int maskIdx)
{
	mask += maskIdx * 81;
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

void CopySudoku(u16* mask, bool* propagated, int idxFrom, int idxTo, int splitIdx, u16 splitValue)
{
	u16* from = mask + 81 * idxFrom;
	u16* to = mask + 81 * idxTo;
	bool* propagatedFrom = propagated + 81 * idxFrom;
	bool* propagatedTo = propagated + 81 * idxTo;

	for (int i = 0; i < 81; i++)
	{
		propagatedTo[i] = propagatedFrom[i];
		if (i == splitIdx)
			to[i] = splitValue;
		else
			to[i] = from[i];
	}
}

void SolveCpu(u16 sudoku[81], u16 result[81])
{
	const int maxMasks = 1000;
	u16 mask[81 * maxMasks];
	std::fill_n(mask, 81 * maxMasks, 0x1ff);

	bool propagated[81 * maxMasks] = { false };

	int activeMasks = 1;
	FillMask(sudoku, mask);

	while (1)
	{
		int activeMasksNew = activeMasks;
		for (int i = 0; i < activeMasks; i++)
		{
			while (PropagatePossibilities(mask, propagated, i)) {}

			int minimumCount = 0;
			int minimumIdx = FindIdexWithLowest(mask, minimumCount, i);

			if (minimumCount == 1) //we got the solution
			{
				int minimumHigherThanOne = -1;
				int minimumHigherIdx = FindIdexWithLowestGreaterThanOne(mask, minimumHigherThanOne, i);

				if (minimumHigherIdx >= 0)
				{
					u16 x = mask[81 * i + minimumHigherIdx];
					bool first = true;
					for (int s = 0; s <= 9; s++)
						if (((x >> s) & 1) == 1)
						{
							CopySudoku(mask, propagated, i, first ? i : activeMasksNew, minimumHigherIdx, 1 << s);
							if (!first)
								activeMasksNew++;
							first = false;
						}
				}
				else
				{
					FillResult(mask, result, i);
					return;
				}
			}
			else //wrong way
			{
			}
		}
		activeMasks = activeMasksNew;
		if (activeMasks > maxMasks)
			std::cout << "Active masks: " << activeMasks << std::endl;
	}
}
