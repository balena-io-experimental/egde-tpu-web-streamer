# Built using https://coral.withgoogle.com/tutorials/accelerator/
FROM balenalib/raspberrypi3-debian

# Add Google Coral sources lists
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install the TPU packages we will need
RUN install_packages libedgetpu1-std \
             libedgetpu-dev \
             python3-edgetpu \
            #  edgetpu-examples \
             python3-pip

# udev in the container to enable TPU correctly
ENV UDEV=1
COPY 99-tpu.rules /etc/udev/rules.d/99-tpu.rules

# # Set our working directory
WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY models/ models/

COPY src/ src/

COPY run.sh run.sh

# Flip the camera vertically
ENV VFLIP=True

CMD ["bash","run.sh"]
