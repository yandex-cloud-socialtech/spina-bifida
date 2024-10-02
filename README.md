# spina-bifida

## Описание проекта
Open source приложение для анализа признаков патологии центральной нервной системы на УЗИ беременных женщин

Приложение для детектирования спектра патологий центральной нервной системы (в том числе Spina bifida) на УЗИ плода в первом триместре беременности. В основе приложения пять моделей, направленных на детекцию области интереса, плоскости, определения качества изображения, а также вероятности целевой патологии. Для корректного анализа необходимо загрузить один ключевой кадр в сагиттальной или в аксиальной плоскостях. 
После получения результата есть возможность сохранять обратную связь.

Совместная разработка:
- [Yandex Cloud](https://yandex.cloud/ru)
- [Школы анализа данных](https://shad.yandex.ru/)
- [Благотворительного фонда «Спина бифида»](https://helpspinabifida.ru/)
- [НМИЦ АКУШЕРСТВА, ГИНЕКОЛОГИИ И ПЕРИНАТОЛОГИИ им. В.И. Кулакова Минздрава России](https://ncagp.ru/).

## Скриншоты
![Screen Shot 2024-08-18 at 20 53 06](https://github.com/user-attachments/assets/3a6913e0-0bb5-4571-8d50-00765bce7b3b)

## Ссылки
[ДЕМО СЕРВИСА](https://spinabifida.cloudtechport.com/)

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

---

© 2024 ООО "Яндекс" / Yandex LLC

Программное обеспечение предоставляется на условиях лицензии - Стандартная общественная лицензия GNU Афферо версии 3.0 / The program is distributed under the terms of the GNU Affero General Public License.

С текстом лицензии можно ознакомиться на сайте / See the text of the License on the website https://www.gnu.org/licenses/.
