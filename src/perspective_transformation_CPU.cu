#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
using namespace std;

#define IMAGE_ROW 1300
#define IMAGE_COL 950

fstream file_coord;
fstream file_image;
fstream file_pt;

struct Image
{
	int R;
	int G;
	int B;
};

void normalize(float *matrix, float *I, float a, int i){
	for (int j = 0; j < 8; j++)
	{
		matrix[i*8+j] /= a;
		I[i*8+j] /= a;
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

void gauss(float *matrix, float *I, int i, int idx){
	for (int j = 0; j < 8; j++)
	{
		if (j!=idx)
		{
			float tmp = matrix[j*8+i];
			for (int k = 0; k <8; k++)
			{
				I[j*8+k] -= I[idx*8+k]*tmp;
				matrix[j*8+k] -= matrix[idx*8+k]*tmp;
			}
		}
	}
}

void homography(float *homo, float *matrix, float *target_coord, int *seq){
	float tmp;
	for (int i = 0; i < 8; i++)
	{
		tmp = 0;
		for (int j = 0; j < 8; j++)
		{
			tmp += matrix[seq[i]*8+j]*target_coord[j];
		}
		homo[i] = tmp;
	}	
	homo[8] = 1;
}

void perspective_trans(float *homo, Image *ori_image, Image *tar_image, float *x, float *y){
	float row_idx, col_idx;
	for (int i = 0; i < IMAGE_ROW; i++)
	{
		for (int j = 0; j < IMAGE_COL; j++)
		{
			row_idx = (homo[0]*j+homo[1]*i+homo[2])/(homo[6]*j+homo[7]*i+1);
			col_idx = (homo[3]*j+homo[4]*i+homo[5])/(homo[6]*j+homo[7]*i+1);
			if (row_idx<IMAGE_COL && col_idx<IMAGE_ROW && row_idx>=0 && col_idx>=0)
			{
				tar_image[int(col_idx)*IMAGE_COL+int(row_idx)].R = ori_image[i*IMAGE_COL+j].R;
				tar_image[int(col_idx)*IMAGE_COL+int(row_idx)].G = ori_image[i*IMAGE_COL+j].G;
				tar_image[int(col_idx)*IMAGE_COL+int(row_idx)].B = ori_image[i*IMAGE_COL+j].B;
			}
		}
	}
}

void output_img(Image *tar_image){
	file_pt.open("ptimage.txt", ios::out);
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
	      float *targ_col_idx, float *I, int *flag, float *coord_matrix,
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
			flag[i] = 0;
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
	double start_time, end_time;

	float* coord_matrix = new float[8*8];
	float* I = new float[8*8];
	float* homomatrix = new float[9];
	int* flag = new int[8];
	int* seq = new int[8];
	int idx;

	float* orig_row_idx = new float[4];
	float* orig_col_idx = new float[4];
	float* targ_row_idx = new float[4];
	float* targ_col_idx = new float[4];
	float* target_coord = new float[8];

	Image* ori_image = new Image[IMAGE_ROW*IMAGE_COL];
	Image* tar_image = new Image[IMAGE_ROW*IMAGE_COL];

	Init(orig_row_idx, orig_col_idx, targ_row_idx, targ_col_idx, I, 
		 flag, coord_matrix, target_coord, ori_image);

	start_time = clock();

	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (coord_matrix[j*8+i]!=0 && flag[j]==0)
			{
				idx = j;
				flag[j] = 1;
				seq[i] = j;
				break;
			}
		}
		float a = coord_matrix[idx*8+i];
		normalize(coord_matrix,I,a,idx);
		gauss(coord_matrix, I, i, idx);
	}

	homography(homomatrix, I, target_coord, seq);

	// print_matrix(I,8,8);
	// print_matrix(homomatrix,3,3);

	perspective_trans(homomatrix, ori_image, tar_image, targ_row_idx, targ_col_idx);
	
	end_time = clock();
 
	cout << "Execution Time: " << (end_time - start_time)/CLOCKS_PER_SEC << endl;

	output_img(tar_image);
	
	return 0;
}
