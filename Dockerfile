FROM tiangolo/uwsgi-nginx-flask:python3.6-alpine3.7
RUN apk --update add bash nano
ENV STATIC_URL /static
ENV STATIC_PATH /usr/lib/cgi-bin/static
COPY ./requirements.txt //usr/lib/cgi-bin/requirements2.txt
RUN pip install -r //usr/lib/cgi-bin/requirements2.txt
