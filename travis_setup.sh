#!/usr/bin/env bash

set -e

mkdir -p $HOME/.mujoco

# Avoid using pyenv in travis, since it adds ~7 minutes to turnaround time
if [ "$TRAVIS_OS_NAME" == "osx" ]
then
    # https://github.com/travis-ci/travis-ci/issues/9640
    sudo softwareupdate --install "Command Line Tools (macOS High Sierra version 10.13) for Xcode-9.4"
    brew update
    brew install open-mpi
    brew install gcc
    brew link --overwrite gcc
    curl $MUJOCO_FOR_OSX | tar xz -C $HOME/.mujoco/
elif [ "$TRAVIS_OS_NAME" == "linux" ]
then
    # Because this is flaky, try several times
    set +e
    COUNT=0
    while [  $COUNT -lt 5 ]; do
       sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
       if [ $? -eq 0 ];then
          break
       fi
       let COUNT=COUNT+1
    done
    if [ $COUNT -ge 5 ]; then
        echo "Failed to download patchelf"
        exit 1
    fi
    set -e

    sudo chmod +x /usr/local/bin/patchelf
    curl $MUJOCO_FOR_LINUX | tar xz -C $HOME/.mujoco/

    sudo apt-get update
    sudo apt-get install -y openmpi-bin libopenmpi-dev libosmesa6-dev libglew-dev
fi
