#include "stereomatch.hpp"
using namespace std;
cv::Mat
bgr_to_grey(const cv::Mat& bgr)
{
    int width = bgr.size().width;
    int height = bgr.size().height;
    cv::Mat grey(height, width, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uchar r = 0.299 * bgr.at<cv::Vec3b>(y, x)[2];
            uchar g = 0.587 * bgr.at<cv::Vec3b>(y, x)[1];
            uchar b = 0.114 * bgr.at<cv::Vec3b>(y, x)[0];
            grey.at<uchar>(y, x) = uchar(r + g + b);
        }
    }

    return grey;
}
int64_t ssd_kernel(const cv::Mat& in1, const cv::Mat& in2,const int size )
{
    int64_t ssd = 0;
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j< size; j++)
        {
            int64_t square_diff = pow((in1.at<uchar>(i,j) - in2.at<uchar>(i,j)),2);
            ssd += square_diff;
        }
    }
    return ssd;
}
cv::Mat ssd(const cv::Mat& limg, const cv::Mat& rimg, std::string type, bool add_constant)
{
    // CHECK_EQ(limg.size().height,rimg.size().height);
    // CHECK_EQ(limg.size().width,rimg.size().width);
    int width = limg.size().width;
    int height = limg.size().height;
    int max_offset = 120;
    int window_size = 7;
    int halfsize = window_size/2;
    // CHECK_EQ(window_size%2 !=0,true);
    cv::Mat left, left_ = bgr_to_grey(limg);
    cv::Mat right, right_ = bgr_to_grey(rimg);
    cv::copyMakeBorder(left_, left,halfsize,halfsize,halfsize,halfsize,CV_HAL_BORDER_CONSTANT,0);
    cv::copyMakeBorder(right_, right,halfsize,halfsize,halfsize,halfsize,CV_HAL_BORDER_CONSTANT,0);
    if (add_constant)
    {
        right += 10;
    }
    cv::Mat disparity(height,width,CV_8U,9999);
    // valid point halfsize--->width/height
    // valid point domain
    for(int i = 0; i < height;i++)//each row
    {
        for(int j = 0; j < width; j++)//each column ie. pixel
        {

            int64_t min_sum_square_diff = numeric_limits<int64_t>::max();
            int min_ssd_index = 0;
            int16_t search_max = min(j + max_offset , width);
            int16_t search_min = max(0, j - max_offset);
            //valid region(padding) domain
            if(type == "left")
            {
                // int roi_sy = i, roi_sx = j, ;
                for(int k = search_min ; k < j ; k++) //right another img's one row each column
                {
                    cv::Mat roi_mat1,roi_mat2;
                    roi_mat1 = left(cv::Range(i, i +window_size),cv::Range(j ,j +window_size));
                    // cv::Mat roi_mat2 = right(roi2);
                    roi_mat2 = right(cv::Range(i , i +window_size),cv::Range(k ,k +window_size));
                    
                    int64_t tmp_ssd =  ssd_kernel(roi_mat1,roi_mat2,window_size);
                    if(tmp_ssd < min_sum_square_diff)
                    {
                        min_sum_square_diff = tmp_ssd;
                        min_ssd_index = k;
                    }
                }

                disparity.at<uchar>(i,j) = max(j- min_ssd_index,1) ;
            }
            else
            {
                for(int k = j ; k < search_max ; k++) //left another img's one row each column
                {
                    cv::Mat roi_mat1,roi_mat2;
                    roi_mat1 = right(cv::Range(i , i +window_size),cv::Range(j ,j +window_size));
                    roi_mat2 = left(cv::Range(i , i +window_size),cv::Range(k ,k +window_size));
                    
                    int64_t tmp_ssd =  ssd_kernel(roi_mat1,roi_mat2,window_size);
                    if(tmp_ssd < min_sum_square_diff)
                    {
                        min_sum_square_diff = tmp_ssd;
                        min_ssd_index = k;
                    }
                }
                disparity.at<uchar>(i,j) = max(min_ssd_index - j,1) ;
            }
        }
    }
    return disparity;
}