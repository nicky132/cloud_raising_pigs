#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

CUR_PATH=$(cd $(dirname $0);pwd)
BASE_PATH=${CUR_PATH}/../..
SDK_PATH=""

# change to Debug if you want
OUR_BUILD_TYPE=Release

# windows or linux
unamestr=`uname`
gitbash=`echo ${unamestr} | grep MINGW64_NT`

toml2unix()
{
  if [ ! -d $1 ];then
    return
  fi

  for file in $(ls $1)
  do
    if [ -d $1"/"$file ];then
      toml2unix $1"/"$file
    elif [ "${file##*.}"x = "toml"x -o "${file##*.}"x = "conf"x ];then
      sed -i 's/\r//g' $1"/"$file
      echo "dos2unix converting file" $1"/"$file " to to Unix format..."
    fi
  done
}

get_sdk_path()
{
  sdk_count=0
  for dir in $(ls ${BASE_PATH})
  do
    if [ -d ${BASE_PATH}"/"$dir ];then
      result=$(echo $dir | grep "modelbox-")
      if [ ! -z "${result}" ];then
        SDK_PATH=${BASE_PATH}"/"$dir
        sdk_count=$(($sdk_count+1))
      fi
  fi
  done

  if [ ${sdk_count} -ne 1 ];then
    echo "err, find ${sdk_count} sdk, exit"
	  read -p "press Enter to continue..." test_enter_key
    exit 1
  fi
}
get_sdk_path

# do c++ build if have c++ flowunit
cplusplus_build()
{
  need_build=0
  for file in $(ls ${CUR_PATH}/flowunit_cpp)
  do
    if [ -d ${CUR_PATH}/flowunit_cpp/$file ];then
      need_build=1
      break
    fi
  done
  if [ "${need_build}" = "0" ];then
    return
  fi

  export CMAKE_INCLUDE_PATH=${SDK_PATH}/include
  export CMAKE_LIBRARY_PATH=${SDK_PATH}/lib

  mkdir -p ${CUR_PATH}/build
  cd ${CUR_PATH}/build

  if [ ! -z ${gitbash} ]; then
    cmake  -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=${OUR_BUILD_TYPE} ..
  else
    toolchain=""
    if [ -e ${SDK_PATH}/toolchain/project_toolchain.cmake ];then
      export TOOLCHAIN_SDK_PATH=${SDK_PATH}
      toolchain=-DCMAKE_TOOLCHAIN_FILE=${SDK_PATH}/toolchain/project_toolchain.cmake
    fi
    cmake  -DCMAKE_BUILD_TYPE=${OUR_BUILD_TYPE} ${toolchain} ..
  fi
  cmake --build . --target install --config ${OUR_BUILD_TYPE} -- -j4
}
cplusplus_build

# avoid windows format toml file, it makes toml error
toml2unix ${CUR_PATH}/graph
toml2unix ${CUR_PATH}/etc
toml2unix ${CUR_PATH}/model
toml2unix ${CUR_PATH}/run/editor
toml2unix ${CUR_PATH}/bin

# create run
mkdir -p ${CUR_PATH}/hilens_data_dir
mkdir -p ${CUR_PATH}/dependence/python
if [ ! -z ${gitbash} ]; then
  REAL_PATH=`cygpath -m -d ${CUR_PATH}`
  MAIN_EXT=".bat"
  PYTHON_CMD=${BASE_PATH}/python-embed/python.exe
else
  REAL_PATH=${CUR_PATH}
  MAIN_EXT=".sh"
  chmod +x ${CUR_PATH}/bin/main${MAIN_EXT}
  chmod +x ${CUR_PATH}/rpm_copyothers.sh
  PYTHON_CMD=python3
fi

# avoid install each time
if ! cmp -s ${REAL_PATH}/dependence/modelbox_requirements.txt ${REAL_PATH}/dependence/python/installed_modelbox_requirements.txt
then
  rm -rf ${REAL_PATH}/dependence/python/*
  echo "start to install modelbox_requirements.txt"
  # install wheel if needed
  for file in $(ls ${REAL_PATH}/dependence/wheel)
  do
    if [ "${file##*.}"x = "whl"x ];then
      ${PYTHON_CMD} -m pip install ${REAL_PATH}/dependence/wheel/$file --target=${REAL_PATH}/dependence/python
    fi
  done
  # install modelbox_requirements.txt
  ${PYTHON_CMD} -m pip install -r ${REAL_PATH}/dependence/modelbox_requirements.txt --target=${REAL_PATH}/dependence/python --timeout=3 --retries=1
  if [ $? -eq 0 ]; then
    cp -rf ${REAL_PATH}/dependence/modelbox_requirements.txt ${REAL_PATH}/dependence/python/installed_modelbox_requirements.txt
  fi
  # del numpy and opencv if not exist in requirements
  if [ `grep -c "numpy" ${REAL_PATH}/dependence/modelbox_requirements.txt` -eq '0' ];then
    rm -rf ${REAL_PATH}/dependence/python/numpy
    rm -rf ${REAL_PATH}/dependence/python/numpy*dist-info
  fi
  if [ `grep -c "opencv_python" ${REAL_PATH}/dependence/modelbox_requirements.txt` -eq '0' ];then
    rm -rf ${REAL_PATH}/dependence/python/cv2
    rm -rf ${REAL_PATH}/dependence/python/opencv_python*dist-info
  fi
fi

# run used define build script
if [ ! -z ${gitbash} ]; then
  if [ -e ${REAL_PATH}/dependence/user_build_win.sh ]; then
    sh ${REAL_PATH}/dependence/user_build_win.sh ${REAL_PATH} ${PYTHON_CMD}
  fi
else
  if [ -e ${REAL_PATH}/dependence/user_build_linux.sh ]; then
    sh ${REAL_PATH}/dependence/user_build_linux.sh ${REAL_PATH} ${PYTHON_CMD}
  fi
fi

echo ""
echo "build success: you can run main"${MAIN_EXT}" in ./bin folder";
echo ""

# wait 5s , it not p press, auto exit
if [ ! -z ${gitbash} ]; then
  KEY_VALUE=""
  echo -e "Press 'p' to pause..., any key to exit\n";
  read -rs -t5 -n1 KEY_VALUE
  if [ "${KEY_VALUE}" = "p" -o "${KEY_VALUE}" = "P" ];then
    read -p "Press any key to exit" test_enter_key
  fi
fi