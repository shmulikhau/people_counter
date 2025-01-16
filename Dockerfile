FROM pytorch/pytorch

WORKDIR /

ADD requirements.txt requirements.txt
ADD requirements-binary.txt requirements-binary.txt

RUN pip install -r requirements.txt
RUN pip install -r requirements-binary.txt --only-binary=:all:

RUN mkdir temp_files
ADD weights weights
ADD models models
ADD service service

EXPOSE 8000

CMD python -m service.service