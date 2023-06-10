#! /bin/bash

# set the default value to be cpu
device_type=cpu;

check_rocm() {
    if command -v rocminfo; then
        echo "ROCm is installed";
        device_type=hip;
    else
        echo "ROCm is not installed";
    fi
}

check_cuda(){
    if command -v nvidia-smi; then
        echo "CUDA is installed"
        device_type=gpu;
    else
        echo "CUDA is not installed";
    fi
}

# Check if CUDA or ROCm is installed
check_cuda;
check_rocm;

echo "The device type is $device_type"

# run the load the data to vectorDB
python3 ingest.py --device_type $device_type

# run the langchain component
python3 cageGPT.py