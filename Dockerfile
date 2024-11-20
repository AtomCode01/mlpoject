FROM python:3.10-slim-buster
WORKDIR /app
COPY ./ app
RUN pip3 install --upgrade pip
RUN apt update -y && apt install awscli -y
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install scikit-learn
RUN pip3 install catboost
RUN pip3 install xgboost
RUN pip3 install dill
RUN pip3 install flask
#RUN pip3 install -r requirements.txt
CMD ["python3", "app.py"]
