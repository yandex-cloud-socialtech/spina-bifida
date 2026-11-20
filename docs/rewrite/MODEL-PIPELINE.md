# ML-пайплайн первого релиза

## Граница системы

Пайплайн намеренно находится внутри backend (`spina_bifida.inference`) и вызывается только отдельным worker. Внешнего plugin API, model discovery и отдельных model-контейнеров нет. HTTP API не загружает веса и не выполняет inference.

## Файлы модели

Для каждой версии `v1` и `v2` обязательны пять файлов:

- `best_object_detection.pt` — YOLOv8 detector плоскости/ROI;
- `axial_quality.pt` и `sagittal_quality.pt` — оценка качества;
- `axial_pathology.pt` и `sagittal_pathology.pt` — оценка признаков.

Всего десять SHA-256 закреплены в `deploy/rewrite/models/manifest.sha256`. Веса [скачиваются отдельно](../../deploy/rewrite/models/README.md) и монтируются в worker как `/models:ro`.

## Последовательность обработки

1. Pillow декодирует кадр и переводит его в RGB.
2. YOLO выбирает первую detection при `conf=0.1` и определяет `sagittal`/`axial`.
3. ROI округляется по Pillow-compatible правилам и ограничивается размерами кадра; пустой или неизвестный ROI даёт штатный rejected-quality результат. Получившийся exact crop и exact ROI box являются входом предсказания.
4. Только exact crop переводится в float tensor, транспонируется как в legacy-процессоре, min-max нормализуется и уменьшается до `255×255` через area interpolation. Quality, assessment и Grad-CAM++ рассчитываются по нему.
5. Для `v1` загружаются DenseNet121, для `v2` — DenseNet201; у `v2` сохраняется дополнительная перестановка осей перед classifier.
6. Quality и assessment берут `sigmoid(output[0][1])`. Текущие пороги обоих решений равны `0.5`.
7. Grad-CAM++ строится по `class_layers.relu`, нормализуется, смешивается с grayscale exact crop палитрой `RdBu` и возвращается в ориентацию исходного ROI.
8. Для отображения exact ROI расширяется на 10% с каждой стороны (`ceil` по ширине и высоте с ограничением границами исходного кадра). Padded crop используется только в PNG assets `roi` и `heatmap`: exact heatmap вставляется в него без растяжения. Padded assets не поступают в классификаторы и не меняют quality/assessment scores, категории или exact ROI box в result contract.
9. Worker сохраняет result contract и PNG assets `original`, `roi`, `heatmap` с SHA-256.

Категории assessment: `no_expressed_signal` и `attention_required`. Score 0–1 — технический выход модели, не диагноз и не калиброванная клиническая вероятность.

## Устройства и воспроизводимость

`device: cpu` использует CPU image с PyTorch CPU. `device: cuda` допустим только после успешной проверки NVIDIA driver, Container Toolkit и `docker run --gpus all`; скрытого fallback на CPU нет. GPU overlay резервирует NVIDIA device только worker.

Golden suite фиксирует v1-результаты четырёх version-controlled примеров с допуском `0.0001`, exact ROI, размерами и SHA-256 exact crop/heatmap до display padding. Canonical checksums снимаются и проверяются в Linux `test-cpu` image на той же runtime-базе, что и production CPU image: это исключает разницу PNG codec и CPU backend между macOS и Linux. Отдельный тест загружает оба полных bundle и проверяет все десять весов.

## Когда выделять модуль

Вынос в отдельный интерфейс обсуждается только после появления второго действительно отличающегося пайплайна или независимого жизненного цикла релизов/ресурсов. До этого расширение остаётся узким изменением `InferenceService`, contracts, manifest и golden tests; неиспользуемая plugin-архитектура не вводится.
