#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

// Modified from thresh.cpp
// static double getThreshVal_Otsu_8u(const Mat& _src)

double otsu_8u_with_mask(const Mat1b src, const Mat1b& mask)
{
    const int N = 256;
    int M = 0;
    int i, j, h[N] = { 0 };
    for (i = 0; i < src.rows; i++)
    {
        const uchar* psrc = src.ptr(i);
        const uchar* pmask = mask.ptr(i);
        for (j = 0; j < src.cols; j++)
        {
            if (pmask[j])
            {
                h[psrc[j]]++;
                ++M;
            }
        }
    }

    double mu = 0, scale = 1. / (M);
    for (i = 0; i < N; i++)
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for (i = 0; i < N; i++)
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i] * scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
            continue;

        mu1 = (mu1 + i*p_i) / q1;
        mu2 = (mu - q1*mu1) / q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if (sigma > max_sigma)
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

double threshold_with_mask(Mat1b& src, Mat1b& dst, double thresh, double maxval, int type, const Mat1b& mask = Mat1b())
{
    if (mask.empty() || (mask.rows == src.rows && mask.cols == src.cols && countNonZero(mask) == src.rows * src.cols))
    {
        // If empty mask, or all-white mask, use cv::threshold
        thresh = cv::threshold(src, dst, thresh, maxval, type);
    }
    else
    {
        // Use mask
        bool use_otsu = (type & THRESH_OTSU) != 0;
        if (use_otsu)
        {
            // If OTSU, get thresh value on mask only
            thresh = otsu_8u_with_mask(src, mask);
            // Remove THRESH_OTSU from type
            type &= THRESH_MASK;
        }

        // Apply cv::threshold on all image
        thresh = cv::threshold(src, dst, thresh, maxval, type);

        // Copy original image on inverted mask
        src.copyTo(dst, ~mask);
    }
    return thresh;
}