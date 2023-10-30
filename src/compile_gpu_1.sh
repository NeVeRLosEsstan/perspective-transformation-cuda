nvcc -c perspective_transformation_GPU_device1.cu -arch=sm_20
g++ -o pt_gpu_device1 perspective_transformation_GPU_device1.o `OcelotConfig -l`
./pt_gpu_device1
