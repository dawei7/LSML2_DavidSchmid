FROM ubuntu:20.04

ENV ROOT_PATH="/app"

ADD requirements.txt requirements.txt
RUN echo installing \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install \
        python3 \
        python3-pip \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
    && pip3 install -r requirements.txt \
    && mkdir -p ${ROOT_PATH}

COPY app.py face_mask_detection_model.h5 ${ROOT_PATH}
COPY templates ${ROOT_PATH}/templates
COPY face_detection ${ROOT_PATH}/face_detection
WORKDIR ${ROOT_PATH}

EXPOSE 5000
ENTRYPOINT ["python3", "app.py"]