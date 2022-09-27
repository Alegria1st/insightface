FROM nvcr.io/nvidia/pytorch:21.11-py3

WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN pip install --no-cache-dir opencv-python==4.4.0.46 --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org
RUN pip install mxnet-cu112 --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org

ADD requirements.txt /app
RUN pip install -r requirements.txt --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org
