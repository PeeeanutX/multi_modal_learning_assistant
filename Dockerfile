FROM python:3.10-slim

ARG USER=appuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USER} \
    && useradd -u ${UID} -g ${GID} -s /bin/bash -m ${USER}

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

USER ${USER}

CMD ["streamlit", "run", "run_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]

LABEL authors="johnto"

ENTRYPOINT ["top", "-b"]