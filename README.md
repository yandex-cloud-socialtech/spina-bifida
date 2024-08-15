# spina-bifida

## Описание проекта
Описание проекта на [сайте]().

## Процесс установки на VM

1. Установить окружение Python. Рекомендуется использовать [miniconda](https://docs.anaconda.com/miniconda/)
2. При необходимости, создать отдельное окружение для проекта
```
conda create -n sb python==3.12
conda activate sb
```
3. Клонировать репозиторий проекта:
```
git clone https://github.com/yandex-cloud-socialtech/spina-bifida
cd spina-bifida
```
4. Установить OpenCV через `conda`:
```
conda install -c conda-forge opencv
```
5. Установить необходимые библиотеки для OpenCV:
```
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```
6. Установить зависимости
```
pip install -r requirements.txt
```
7. Записать необходимые ключи для доступа к Object Storage для сохранения результатов в файл `.env`:
```
ACCESS_KEY=...
SECRET_KEY=...
BUCKET=...
```
8. Скачать [модели](https://storage.yandexcloud.net/spina-bifida-models/models.zip) и скопировать их в папку `models`
9. Запустить приложение `screen`, чтобы выполнение приложения не прерывалось после прекращения сеанса.
10. Запустить приложение:
```
streamlit run app.py
```
