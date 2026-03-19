FROM registry-sc-harbor.amd.com/rocm-ci-images/compute-rocm-dkms-no-npi-hipclang:16993-ubuntu-24.04

SHELL [ "/bin/bash", "-c" ]

COPY AMD_CA.crt /usr/local/share/ca-certificates

RUN DEBIAN_FRONTEND=noninteractive  \
    update-ca-certificates          \
    && apt update -y                \
    && apt upgrade -y               \
    && apt install -y               \
        sudo                        \
        nano                        \
        locales                     \
        ninja-build                 \
    && locale-gen en_US.UTF-8       \
    && dpkg-reconfigure locales     \
    && apt install -y               \
        software-properties-common  \
        jq                          \
        git                         \
        git-lfs                     \
        wget                        \
        curl                        \
        build-essential             \
        g++                         \
        # gcovr                       \
        # libasan8                    \
        # libtsan2                    \
        git-core                    \
        rccl-dev                    \
        libdw-dev                   \
        python3-pip                 \
        libgtest-dev                \
        libgmock-dev                \
        inotify-tools               \
        # dotnet-runtime-8.0          \
        # dotnet-sdk-8.0              \
        bash-completion             \
        # linux-tools-generic         \
        # xfonts-base                 \
        # xfonts-75dpi                \
        # xfonts-100dpi               \
        # xfonts-utils                \
        # xfonts-encodings            \
        # libfontconfig               \
        python-is-python3           \
    && echo "Done"

# apt install rocm-llvm-dev takes till heat death of universe
# to download for some reason, download manually and install
RUN echo "Downloading rocm-llvm-dev (this may take more than 10 mins) ..." \
    && RLD_DOWNLOAD_URL=$(echo $(apt download --print-uris rocm-llvm-dev | awk '{print $1}') | sed "s/'//g") \
    && aria2c -x 16 $RLD_DOWNLOAD_URL -o rocm-llvm-dev.deb \
    && dpkg --force-overwrite -i rocm-llvm-dev.deb \
    && rm rocm-llvm-dev.deb \
    && echo "Installed rocm-llvm-dev"

ARG CLANGD_VER=21.1.8
ARG CLANGD_URL="https://github.com/clangd/clangd/releases/download/${CLANGD_VER}/clangd-linux-${CLANGD_VER}.zip"
RUN echo "Installing clangd to /usr"        \
    && wget ${CLANGD_URL} -O clangd.tar.xz  \
    && unzip clangd.tar.xz                  \
    && rsync -a clangd_${CLANGD_VER}/  /usr        \
    && rm -rf clangd*                       \
    && echo "Done"

RUN if [[ -e /usr/local/bin/python3.8 ]]; then  \
    rm -f "/usr/local/bin/python3.8";           \
    fi;                                         \
    if [[ -e /usr/local/bin/git ]]; then        \
    rm -f "/usr/local/bin/git";                 \
    fi;

ARG WORKDIR=/home/mkuriche/work/mi300-df-pr-container
WORKDIR ${WORKDIR}

ARG ROCM_INTERNAL_DIR=rocm-systems                 # set this to copy from host, relative to the docker build context
ARG ROCM_SYSTEMS_DIR=${WORKDIR}/rocm-systems

COPY ${ROCM_INTERNAL_DIR}   ${ROCM_SYSTEMS_DIR}
COPY zbuild.sh              ${ROCM_INTERNAL_DIR}
COPY remove.sh              ${ROCM_SYSTEMS_DIR}
COPY ptl-disable.patch      ${ROCM_SYSTEMS_DIR}

RUN echo "Building rocm-systems"                            \
    && cd ${ROCM_SYSTEMS_DIR}                               \
    && git checkout users/mkuriche/umc-df-ipdiscovery       \
    && echo "git status:"                                   \
    && git status                                           \
    && git reset --hard                                     \
    && git apply ptl-disable.patch                          \
    && ./remove.sh                                          \
    && bash -c ${ROCM_SYSTEMS_DIR}/zbuild.sh                \
    && echo "Done"

RUN echo "Building babel stream"                        \
    && git clone https://github.com/UoB-HPC/BabelStream.git babel-stream \
    && pushd babel-stream                                       \
    && git checkout 8ab508851abad6be983108019d9453682880a366    \
    && cmake \
        -B build    \
        -DMODEL=hip             \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm            \
        -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc      \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc    \
        -S . \
    && cmake --build build \
    && cmake --install build \
    && popd \
    && echo "Done"
