FROM python:3.7

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install libopenmpi-dev -y

ARG USERNAME=container
ARG USER_UID=1008
ARG USER_GID=1009
ARG WORK_DIR=/home/$USERNAME

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --create-home --shell /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
WORKDIR $WORK_DIR

COPY --chown=$USERNAME . . 

RUN pip install -e .

#fix for openmpi err( Read -1, expected <someNumber>, errno =1)
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none 

CMD [ "python", $WORKDIR/test/test_ppo.py ]


