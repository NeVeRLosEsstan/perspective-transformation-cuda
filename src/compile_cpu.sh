nvcc -c perspective_transformation_CPU.cu -arch=sm_20
g++ -o pt_cpu perspective_transformation_CPU.o `OcelotConfig -l`
./pt_cpu
