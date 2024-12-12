# Frontend/Dockerfile

FROM python:3.10.8

RUN mkdir -p app/frontend

COPY Streamlit_Frontend app/frontend

WORKDIR /app/frontend

RUN pip install --upgrade pip
RUN pip install opencv-python-headless opencv-python

RUN pip install -r requirements.txt

EXPOSE 8501

EXPOSE 8502

ENTRYPOINT ["streamlit","run"]

CMD ["Hello.py"]
