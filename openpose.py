import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
project_name = splitext(basename(git_repo_url))[0]
print(project_name)


if not exists(project_name):
  # see: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949
  # install new CMake becaue of CUDA10
  # os.system("wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz")
  print("done 1")
  
  # os.system("tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local")
  print("done 2")
  # clone openpose
  os.system(f"git clone -q --depth 1 {git_repo_url}")
  print("done 3")

  os.system("sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt")
  print("done 4")
  
  # install system dependencies
  os.system("apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev")
  print("done 4")
  
  # install python dependencies
  os.system("pip install -q youtube-dl")
  print("done 5")
  
  # build openpose
  # os.system("cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`")
  print("done 6")
