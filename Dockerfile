FROM python:3.10.0-slim-buster

ENV APP_HOME=/app
RUN mkdir $APP_HOME 
WORKDIR $APP_HOME


LABEL maintainer="TechWithRay"
LABEL version="0.0.1"
LABEL description="cageGPT: Your own LLM to providing protection and confinement for your data"

ENV PYTHONDONOTWRITEBYTECODE 1

RUN apt-get update && apt-get install -y \
    build-essential \
    gettext \
    git \
    cmake \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY ./requirements.txt $APP_HOME/requirements.txt

COPY ./local_run.sh $APP_HOME/local_run.sh

RUN pip3 install -r requirements.txt

CMD ["bash", "local_run.sh"]
