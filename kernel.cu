
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include <algorithm>

__global__ void gaussian_blur(const unsigned char* const inputChannel, unsigned char* const outputChannel, int numRows, int numCols)
{
	int absolute_image_position_x = threadIdx.x + blockIdx.x * blockDim.x;
	int absolute_image_position_y = threadIdx.y + blockIdx.y * blockDim.y;
	int index;

	float color = 0.0f;

	if(absolute_image_position_x >= numCols || absolute_image_position_y >= numRows)
	{
		return;
	}
	index = absolute_image_position_x + absolute_image_position_y * numCols;
	outputChannel[index] = color;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	// TODO
	//
	// NOTE: Be careful not to try to access memory that is outside the bounds of
	// the image. You'll want code that performs the following check before accessing
	// GPU memory:
	//

	int absolute_image_position_x = threadIdx.x + blockIdx.x * blockDim.x;
	int absolute_image_position_y = threadIdx.y + blockIdx.y * blockDim.y;
	int index;
	if (absolute_image_position_x >= numCols || absolute_image_position_y >= numRows) {
		return;
	}
	else {
		index = absolute_image_position_x + numCols*absolute_image_position_y;
		//uchar4 temp = inputImageRGBA[index];
		redChannel[index] = inputImageRGBA[index].x;
		greenChannel[index] = inputImageRGBA[index].y;
		blueChannel[index] = inputImageRGBA[index].z;
	}
}

__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage)
{
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char)*numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char)*numRowsImage*numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char)*numRowsImage* numColsImage));
}

void your_gaussian_blur(const uchar4* const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
	unsigned char *d_redOut, unsigned char *d_greenOut, unsigned char *d_blueOut)
{
	const dim3 blockSize(32, 32);
	const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);
	separateChannels << <gridSize, blockSize >> >(d_inputImageRGBA, numRows, numCols, d_red, d_blue, d_green);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	gaussian_blur << <gridSize, blockSize >> >(d_red, d_redOut, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	gaussian_blur << <gridSize, blockSize >> >(d_blue, d_blueOut, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	gaussian_blur << <gridSize, blockSize >> >(d_green, d_greenOut, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	recombineChannels << <gridSize, blockSize >> >(d_redOut, d_blueOut, d_greenOut, d_outputImageRGBA, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;
int main(int argc, char** argv)
{
	uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
	unsigned char *d_redOut, *d_blueOut, *d_greenOut;

	std::string input_file = "D:\\Documents\\Images\\cpature.jpg";
	std::string output_file = "D:\\Documents\\Images\\capture2.jpg";

	cv::Mat image = cv::imread(input_file, CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Could noto open file: " << input_file << std::endl;
		exit(1);
	}
	cv::cvtColor(image, imageInputRGBA, CV_BGR2BGRA);
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous())
	{
		std::cerr << "Images areen't continuous !! Exiting." << std::endl;
		exit(1);
	}
	h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = image.cols * image.rows;
	checkCudaErrors(cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) *numPixels));
	checkCudaErrors(cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(d_outputImageRGBA, 0, numPixels*sizeof(uchar4));
	checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4)* numPixels, cudaMemcpyHostToDevice));
	d_inputImageRGBA__ = d_inputImageRGBA;
	d_outputImageRGBA__ = d_outputImageRGBA;
	checkCudaErrors(cudaMalloc(&d_redOut, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(&d_greenOut, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(&d_blueOut, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_redOut, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_greenOut, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_blueOut, 0, sizeof(unsigned char) * numPixels));

	allocateMemoryAndCopyToGPU(image.cols, image.rows);
	size_t cols = image.cols;
	size_t rows = image.rows;
	your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, rows, cols, d_redOut, d_blueOut, d_greenOut);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
	cv::Mat output(image.rows, image.cols, CV_8UC4, (void*)h_outputImageRGBA);
	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
	cv::imwrite(output_file, imageOutputBGR);
	checkCudaErrors(cudaFree(d_redOut));
	checkCudaErrors(cudaFree(d_greenOut));
	checkCudaErrors(cudaFree(d_blueOut));
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
}