FROM python:3.9-slim
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"
USER root

# Add user
RUN useradd -ms /bin/bash test_user

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy scripts, source files and tests
COPY . /app/
RUN chmod 777 /app/ /app/data/ /app/tests/

# Entrypoint
USER test_user
WORKDIR /app
ENTRYPOINT /bin/bash