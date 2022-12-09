FROM python:3.8

WORKDIR /workspace
ADD . /workspace

RUN pip install --no-cache-dir -r requirements_django.txt


RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace



EXPOSE 8000

# Here we can keep the command because there is one dockerfile needed.  
CMD python3 manage.py runserver 0.0.0.0:8000