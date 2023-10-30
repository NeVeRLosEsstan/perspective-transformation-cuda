#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include "IndexSave.h"
#include <iomanip>
using namespace std;

#define IMAGE_ROW 1300
#define IMAGE_COL 950
#define BLOCK_SIZE 16

fstream file_coord;
fstream file_image;
fstream file_pt;

struct Image
{
	int R;
	int G;
	int B;
};

__global__ void normalize(float *matrix, float *I, int i){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	float a = matrix[i*8+i];
	if (blockIdx.x==i)
	{
		matrix[idx] /= a;
		I[idx] /= a;
	}
}

__global__ void gauss(float *matrix, float *I, int i){
	if (blockIdx.x!=i)
	{
		I[blockIdx.x*8+threadIdx.x] -= I[i*8+threadIdx.x]*matrix[blockIdx.x*8+i];
		if (threadIdx.x!=i)
		{
			matrix[blockIdx.x*8+threadIdx.x] -= matrix[i*8+threadIdx.x]*matrix[blockIdx.x*8+i];
		}
	}
}

__global__ void swap(float *matrix, float *I, int i){
	float tmp;
	for (int j = i; j < 8; j++)
	{
		if (matrix[i*8+i]==0 && matrix[j*8+i]!=0)
		{
			tmp = matrix[i*8+threadIdx.x];
			matrix[i*8+threadIdx.x] = matrix[j*8+threadIdx.x];
			matrix[j*8+threadIdx.x] = tmp;

			tmp = I[i*8+threadIdx.x];
			I[i*8+threadIdx.x] = I[j*8+threadIdx.x];
			I[j*8+threadIdx.x] = tmp;
		}
	}
}

__global__ void homography(float *homo, float *matrix, float *target_coord){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	float tmp = 0;
	if (idx<8)
	{
		for (int i = 0; i < 8; i++)
		{
			tmp += matrix[idx*8+i]*target_coord[i];
		}
		homo[idx] = tmp;
	}

	homo[8] = 1;
}

__global__ void perspective_trans(float *homo, Image *ori_image, Image *tar_image){
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col<IMAGE_COL && row<IMAGE_ROW)
	{
		float row_idx, col_idx;
	
		row_idx = (homo[0]*col+homo[1]*row+homo[2])/(homo[6]*col+homo[7]*row+1);
		col_idx = (homo[3]*col+homo[4]*row+homo[5])/(homo[6]*col+homo[7]*row+1);
	
		if (row_idx<IMAGE_COL && col_idx<IMAGE_ROW && row_idx>=0 && col_idx>=0)
		{
			tar_image[int(col_idx)*IMAGE_COL+int(row_idx)].R = ori_image[row*IMAGE_COL+col].R;
			tar_image[int(col_idx)*IMAGE_COL+int(row_idx)].G = ori_image[row*IMAGE_COL+col].G;
			tar_image[int(col_idx)*IMAGE_COL+int(row_idx)].B = ori_image[row*IMAGE_COL+col].B;
		}
	}
}

void print_matrix(float *matrix, int row, int col){
	for (int i = 0; i < row; i++)
	{	
		for (int j = 0; j < col; j++)
		{
			cout << matrix[i*row+j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void output_img(Image *tar_image){
	file_pt.open("ptimage_gpu.txt", ios::out);
	for (int i = 0; i < IMAGE_ROW ; i++)
	{
		for (int j = 0; j < IMAGE_COL; j++)
		{
			file_pt << tar_image[i*IMAGE_COL+j].B << " "
					<< tar_image[i*IMAGE_COL+j].G << " "
					<< tar_image[i*IMAGE_COL+j].R << " "
					<< '\n';
		}
	}
}

void Init(float *orig_row_idx, float *orig_col_idx, float *targ_row_idx, 
	      float *targ_col_idx, float *I, float *coord_matrix,
	      float *target_coord, Image *ori_image){

	file_coord.open("coord.txt", ios::in);
	file_image.open("image.txt", ios::in);

	for (int i = 0; i < 4; i++)
	{
		file_coord >> orig_row_idx[i];
		file_coord >> orig_col_idx[i];
	}

	for (int i = 0; i < 4; i++)
	{
		file_coord >> targ_row_idx[i];
		file_coord >> targ_col_idx[i];
	}

	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (i==j)
			{
				I[i*8+j] = 1;
			}
			else I[i*8+j] = 0;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		coord_matrix[2*i*8]    = orig_row_idx[i];
		coord_matrix[2*i*8+1]  = orig_col_idx[i];
		coord_matrix[2*i*8+2]  = 1;
		coord_matrix[2*i*8+3]  = 0;
		coord_matrix[2*i*8+4]  = 0;
		coord_matrix[2*i*8+5]  = 0;
		coord_matrix[2*i*8+6]  = -orig_row_idx[i]*targ_row_idx[i];
		coord_matrix[2*i*8+7]  = -orig_col_idx[i]*targ_row_idx[i];
		coord_matrix[2*i*8+8]  = 0;
		coord_matrix[2*i*8+9]  = 0;
		coord_matrix[2*i*8+10] = 0;
		coord_matrix[2*i*8+11] = orig_row_idx[i];
		coord_matrix[2*i*8+12] = orig_col_idx[i];
		coord_matrix[2*i*8+13] = 1;
		coord_matrix[2*i*8+14] = -orig_row_idx[i]*targ_col_idx[i];
		coord_matrix[2*i*8+15] = -orig_col_idx[i]*targ_col_idx[i];
		target_coord[2*i] = targ_row_idx[i];
		target_coord[2*i+1] = targ_col_idx[i];
	}

	for (int i = 0; i < IMAGE_ROW; i++)
	{
		for (int j = 0; j < IMAGE_COL; j++)
		{
			file_image >> ori_image[i*IMAGE_COL+j].B;
			file_image >> ori_image[i*IMAGE_COL+j].G;
			file_image >> ori_image[i*IMAGE_COL+j].R;
		}
	}
}

int main()
{
	float elapsedTime;
	cudaEvent_t start,stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	float* coord_matrix = new float[8*8];
	float* I = new float[8*8];
	float* homomatrix = new float[9];
	float* d_coord_matrix;
	float* d_I;
	float* d_homomatrix;

	float* orig_row_idx = new float[4];
	float* orig_col_idx = new float[4];
	float* targ_row_idx = new float[4];
	float* targ_col_idx = new float[4];
	float* target_coord = new float[8];
	float* d_target_coord;

	Image* ori_image = new Image[IMAGE_ROW*IMAGE_COL];
	Image* tar_image = new Image[IMAGE_ROW*IMAGE_COL];
	Image* d_ori_image;
	Image* d_tar_image;

	Init(orig_row_idx, orig_col_idx, targ_row_idx, targ_col_idx, I, 
		 coord_matrix, target_coord, ori_image);

	cudaEventRecord(start, 0);

	cudaMalloc((void**)& d_coord_matrix, 8*8 * sizeof(float));
	cudaMalloc((void**)& d_I, 8*8 * sizeof(float));

	cudaMalloc((void**)& d_homomatrix, 9 * sizeof(float));
	cudaMalloc((void**)& d_target_coord, 8 * sizeof(float));
	cudaMalloc((void**)& d_ori_image, IMAGE_ROW*IMAGE_COL * sizeof(Image));
	cudaMalloc((void**)& d_tar_image, IMAGE_ROW*IMAGE_COL * sizeof(Image));

	cudaMemcpy(d_coord_matrix, coord_matrix, 8*8 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, I, 8*8 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_target_coord, target_coord, 8 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ori_image, ori_image, IMAGE_ROW*IMAGE_COL * sizeof(Image), cudaMemcpyHostToDevice);
	
	
	for (int i = 0; i < 8; i++)
	{
		dim3 dimBlock(8);
		dim3 dimGrid(8);
		swap << < dimGrid, dimBlock >> > (d_coord_matrix, d_I, i);
		normalize << < dimGrid, dimBlock >> > (d_coord_matrix, d_I, i);
		gauss << < dimGrid, dimBlock >> > (d_coord_matrix, d_I, i);
	}

	dim3 homodimBlock(8);
	dim3 homodimGrid(1);
	homography << < homodimGrid, homodimBlock >> > (d_homomatrix, d_I, d_target_coord/*, dInd*/);
	
	dim3 ptdimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 ptdimGrid((IMAGE_COL + BLOCK_SIZE - 1)/BLOCK_SIZE, (IMAGE_ROW + BLOCK_SIZE - 1)/BLOCK_SIZE);
	perspective_trans << < ptdimGrid, ptdimBlock >> > (d_homomatrix, d_ori_image, d_tar_image);

	cudaDeviceSynchronize();
	cudaMemcpy(tar_image, d_tar_image, IMAGE_ROW*IMAGE_COL * sizeof(Image), cudaMemcpyDeviceToHost);
	
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Execution Time: " << (elapsedTime)/CLOCKS_PER_SEC << endl;

	output_img(tar_image);

	return 0;
}
