FROM continuumio/anaconda3

RUN pip install pandas keras tensorflow numpy

ADD . /

ENTRYPOINT ["/run.sh"]
