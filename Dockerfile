FROM python:3.7

WORKDIR /spinningup
COPY . .

RUN apt-get update && apt-get install libopenmpi-dev -y\
    && pip install -e .

ARG USERNAME=container
ARG USER_UID=1008
ARG USER_GID=1009

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --create-home --shell /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME

CMD [ "python", $WORKDIR/test/test_ppo.py ]


