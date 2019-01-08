FROM tensorflow/tensorflow:1.7.0-devel-gpu-py3


RUN apt-get update
RUN apt-get install -y vim cython python-numpy python-dev cmake zlib1g-dev \
                            libjpeg-dev xvfb libav-tools xorg-dev python-opengl \
                            libboost-all-dev libsdl2-dev swig
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

jupyter contrib nbextension install --user
