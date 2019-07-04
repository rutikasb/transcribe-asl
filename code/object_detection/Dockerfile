FROM python:2.7
ADD obj_detect.py /home
COPY models /home/models
WORKDIR /home
COPY requirements.txt /home
RUN pip install -r /home/requirements.txt
