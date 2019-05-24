FROM python:2.7

WORKDIR /usr/src/app

VOLUME /usr/src/app/data

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","-u", "xg_boost_tunning.py"]
