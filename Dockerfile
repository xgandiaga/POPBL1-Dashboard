FROM python:3.6
ADD ./requirements.txt /app/requirements.txt
WORKDIR /app
COPY interfaz.py /app
COPY assets /app/assets
COPY work_data_prepro_all.csv /app
COPY work_data_prepro.csv /app
RUN pip install -r requirements.txt
EXPOSE 8050
CMD python /app/interfaz.py