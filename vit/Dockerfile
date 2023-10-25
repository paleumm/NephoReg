FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y python3

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

WORKDIR /app

COPY requirements.txt ./requirements.txt 
RUN pip install -r requirements.txt
RUN pip install timm==0.9.7

COPY new_vit_pseudo_2.pkl ./new_vit_pseudo_2.pkl
COPY app.py ./app.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]