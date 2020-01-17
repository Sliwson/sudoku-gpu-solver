
extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

void computeGold(float *reference, float *idata, const unsigned int len)
{
    const float f_len = static_cast<float>(len);

    for (unsigned int i = 0; i < len; ++i)
    {
        reference[i] = idata[i] * f_len;
    }
}

