# Built using https://coral.withgoogle.com/tutorials/accelerator/
FROM balenalib/raspberrypi3-debian

# Install some utilities we will need
RUN apt-get update && apt-get install build-essential wget feh

# Set our working directory
WORKDIR /usr/src/app

# Need udev for some dynamic dev nodes
ENV UDEV=1

# Fetch latest edge TPU libs
RUN wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz -O edgetpu_api.tar.gz --trust-server-names && \
    tar xzf edgetpu_api.tar.gz && rm edgetpu_api.tar.gz

WORKDIR /usr/src/app/edgetpu_api

# Override the MODEL variable so we can build in a container.
ENV MODEL="Raspberry Pi 3 Model B Rev"
RUN sed -i "s|MODEL=|#MODEL=|g" install.sh

# Pass N to the prompt in the install script if we want to overclock the tpu
RUN yes n | ./install.sh

# Set our working directory
WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY models/ models/

COPY src/ src/

COPY run.sh run.sh

# Flip the camera vertically
ENV VFLIP=True

CMD ["bash","run.sh"]
