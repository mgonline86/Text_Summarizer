FROM python:3.8

RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz graphviz-dev

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

ADD static /app/static/
ADD templates /app/templates/
COPY app.py /app/
COPY summarize.py /app/
COPY word2vec_v_2.py /app/

RUN python -m nltk.downloader stopwords punkt

CMD [ "python", "app.py" ]
