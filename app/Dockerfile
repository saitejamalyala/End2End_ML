FROM python:3.7-slim
EXPOSE 8051
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run stepstoneapp.py