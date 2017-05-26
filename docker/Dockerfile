FROM kaggle/python

WORKDIR /home
RUN git clone https://83a0ca722df6a0647f1ea08fb5df089fbc0ee328:x-oauth-basic@github.com/flowlight0/QuoraQuestionPairs.git

WORKDIR /home/QuoraQuestionPairs
RUN git clone --recursive https://github.com/Microsoft/LightGBM &&\
    cd LightGBM && \
    mkdir build &&\
    cd build &&\
    cmake ..  &&\
    make -j
RUN cd LightGBM/python-package &&  python setup.py install
RUN pip install -r required_packages.txt

    


