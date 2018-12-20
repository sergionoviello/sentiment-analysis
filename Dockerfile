FROM gw000/keras:2.1.4-py3-tf-cpu

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    python-matplotlib \
    python-pillow

RUN apt-get install -y python-pip python-dev build-essential
RUN pip install --upgrade pip

RUN pip install flask \
    tensorflow --upgrade

RUN  python3 -m pip install numpy \
     scipy \
     nltk \
     scikit-learn \
     gensim \
     jupyter \
     matplotlib \
     pandas \
     seaborn \
     ipykernel

RUN python3 -m ipykernel install --user

VOLUME /app
WORKDIR /app

EXPOSE 8888

COPY . /app
WORKDIR /app

# ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
ENTRYPOINT ["/bin/bash"]