# Веса моделей Spina Bifida

Для запуска worker нужны оба комплекта. Команда из корня репозитория скачивает десять
файлов и сверяет их с [манифестом SHA-256](manifest.sha256).

~~~bash
bash deploy/rewrite/download-models.sh "$HOME/spina-bifida-models"
~~~

| Модель | v1 | v2 |
| --- | --- | --- |
| Плоскость и ROI | [best_object_detection.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v1/best_object_detection.pt) | [best_object_detection.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v2/best_object_detection.pt) |
| Качество, аксиальная плоскость | [axial_quality.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v1/axial_quality.pt) | [axial_quality.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v2/axial_quality.pt) |
| Признаки патологии, аксиальная плоскость | [axial_pathology.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v1/axial_pathology.pt) | [axial_pathology.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v2/axial_pathology.pt) |
| Качество, сагиттальная плоскость | [sagittal_quality.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v1/sagittal_quality.pt) | [sagittal_quality.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v2/sagittal_quality.pt) |
| Признаки патологии, сагиттальная плоскость | [sagittal_pathology.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v1/sagittal_pathology.pt) | [sagittal_pathology.pt](https://storage.yandexcloud.net/spina-bifida-models/rewrite/edd86faffde0/v2/sagittal_pathology.pt) |

Сохраняйте файлы в подкаталоги **v1/** и **v2/** соответственно. Для размещения
на сервере используйте [руководство по установке](../../../docs/rewrite/OPERATIONS.md#3-разместить-модели).
