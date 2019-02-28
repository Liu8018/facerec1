#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "opencv2/highgui.hpp"

#ifndef SIMD_OPENCV_ENABLE
#define SIMD_OPENCV_ENABLE
#endif

#include "Simd/SimdDetection.hpp"
#include "Simd/SimdDrawing.hpp"

typedef Simd::Detection<Simd::Allocator> SimdDetection;

#endif // FACEDETECTOR_H
