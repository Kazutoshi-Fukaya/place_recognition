#include "bow/descriptors_manipulator/descriptors_manipulator.h"

using namespace place_recognition;

void DescriptorsManipulator::mean_value(std::vector<cv::Mat>& descriptors,cv::Mat &mean)
{
	if(descriptors.empty()) return;
	
	if(descriptors.size() == 1){
		mean = descriptors[0].clone();
		return;
    }
    
	//binary descriptor
    if (descriptors[0].type() == CV_8U ){
		// determine number of bytes of the binary descriptor
		int L = get_desc_size_bytes(descriptors[0]);
        std::vector<int> sum(L*8,0);
        for(size_t i = 0; i < descriptors.size(); i++){
            const cv::Mat &d = descriptors[i];
            const unsigned char *p = d.ptr<unsigned char>();

            for(int j = 0; j < d.cols; j++, p++){
                if(*p & (1 << 7)) ++sum[j*8    ];
                if(*p & (1 << 6)) ++sum[j*8 + 1];
                if(*p & (1 << 5)) ++sum[j*8 + 2];
                if(*p & (1 << 4)) ++sum[j*8 + 3];
                if(*p & (1 << 3)) ++sum[j*8 + 4];
                if(*p & (1 << 2)) ++sum[j*8 + 5];
                if(*p & (1 << 1)) ++sum[j*8 + 6];
                if(*p & (1))      ++sum[j*8 + 7];
            }
        }

        mean = cv::Mat::zeros(1,L,CV_8U);
        unsigned char *p = mean.ptr<unsigned char>();

        const int N2 = (int)descriptors.size()/2 + descriptors.size()%2;
        for(size_t i = 0; i < sum.size(); i++){
            if(sum[i] >= N2){
                // set bit
                *p |= 1 << (7 - (i % 8));
            }

            if(i % 8 == 7) p++;
        }
    }
    //non binary descriptor
    else{
        assert(descriptors[0].type() == CV_32F);

        mean.create(1,descriptors[0].cols,descriptors[0].type());
        mean.setTo(cv::Scalar::all(0));
        float inv_s = 1.0/double(descriptors.size());
        for(size_t i = 0; i < descriptors.size(); i++) mean +=  descriptors[i] * inv_s;
    }
}

double DescriptorsManipulator::distance(cv::Mat& a,cv::Mat& b)
{
	//binary descriptor
	if(a.type()==CV_8U){
		const uint64_t *pa, *pb;
		pa = a.ptr<uint64_t>();
        pb = b.ptr<uint64_t>();
		
		uint64_t v, ret = 0;
		for(size_t i = 0; i < a.cols/sizeof(uint64_t); i++, pa++, pb++){
			v = *pa ^ *pb;
			v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
			v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) & (uint64_t)~(uint64_t)0/15*3);
			v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
			ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
        }
		return ret;
    }
    else{
		double sqd = 0.;
        assert(a.type() == CV_32F);
        assert(a.rows == 1);
        const float *a_ptr = a.ptr<float>(0);
        const float *b_ptr = b.ptr<float>(0);
        for(int i = 0; i < a.cols; i++){
			sqd += (a_ptr[i] - b_ptr[i])*(a_ptr[i] - b_ptr[i]);
		}
		return sqd;
    }
}

uint32_t DescriptorsManipulator::distance_8uc1(cv::Mat& a,cv::Mat& b)
{
	// binary descriptors
	uint64_t *pa, *pb;
	pa = a.ptr<uint64_t>();
	pb = b.ptr<uint64_t>();

	uint64_t v, ret = 0;
	int n = a.cols/sizeof(uint64_t);
	for(size_t i = 0; i < n; i++, pa++, pb++){
		v = *pa^*pb;
		v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
		v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) & (uint64_t)~(uint64_t)0/15*3);
		v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
		ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
    }
	return ret;
}

size_t DescriptorsManipulator::get_desc_size_bytes(cv::Mat& d) { return d.cols*d.elemSize(); }