FROM continuumio/anaconda3

RUN pip install pandas keras tensorflow numpy joblib

ADD . /

ENTRYPOINT ["/run.sh"]
