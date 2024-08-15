# project-spina-bifida

## Процесс установки на VM

1. Установить окружение Python. Рекомендуется использовать [miniconda](https://docs.anaconda.com/miniconda/)
1. При необходимости, создать отдельное окружение для проекта
```
conda create -n sb python==3.12
conda activate sb
```
1. Клонировать репозиторий проекта:
```
git clone https://github.com/yandex-datasphere/project-spina-bifida
cd project-spina-bifida
```
1. Установить OpenCV через `conda`:
```
conda install -c conda-forge opencv
```
1. Установить необходимые библиотеки для OpenCV:
```
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```
1. Установить зависимости
```
pip install -r requirements.txt
```
1. Записать необходимые ключи для доступа к Object Storage для сохранения результатов в файл `.env`:
```
ACCESS_KEY=...
SECRET_KEY=...
BUCKET=...
```
1. Запустить приложение `screen`, чтобы выполнение приложения не прерывалось после прекращения сеанса.
1. Запустить приложение:
```
streamlit run app.py
```
