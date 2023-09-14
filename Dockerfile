FROM quay.io/kiegroup/kogito-swf-devmode:1.44

USER root

RUN microdnf install -y --nodocs python311 gcc python3.11-devel mesa-libGLU && \
    microdnf clean all && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python

USER 1001

ENV KOGITO_VERSION=1.44.1.Final PATH="${PATH}:/home/kogito/.local/bin" PYTHONPATH="/home/kogito/.local/bin"

COPY requirements.txt /home/kogito/serverless-workflow-project/

RUN pip install numpy --target /home/kogito/.local/bin
RUN pip install -r requirements.txt --target /home/kogito/.local/bin

ENV QUARKUS_EXTENSIONS="org.kie.kogito:kogito-addons-quarkus-serverless-workflow-python:${KOGITO_VERSION}"

CMD ["/home/kogito/launch/run-app-devmode.sh"]