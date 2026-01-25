FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR=/home/submission
# ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR=/home/logs
# ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR=/home/code
# ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR=/home/agent
# ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements_agent.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} conda install -y \
        -c pytorch \
        -c conda-forge \
        faiss-cpu=1.13.2 && \
    # 2. 再运行 Pip 安装 requirements.txt
    # 注意：这里 pip 会检测到 conda 已经安装的 numpy，并基于此安装兼容的 sklearn
    conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# put all the agent files in the expected location
ENV HF_ENDPOINT=https://hf-mirror.com
ENV MODEL_SAVE_PATH=${AGENT_DIR}/embedding-models/bge-m3
RUN mkdir -p ${MODEL_SAVE_PATH}
COPY scripts/download_model.py ${AGENT_DIR}/scripts/download_model.py
RUN conda run -n ${CONDA_ENV_NAME} python ${AGENT_DIR}/scripts/download_model.py && \
    chmod -R 555 ${MODEL_SAVE_PATH}

COPY . ${AGENT_DIR}
