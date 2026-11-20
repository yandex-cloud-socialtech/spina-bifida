# Проверка установки и релизного комплекта

Этот документ описывает воспроизводимые команды и критерии проверки. Он не подтверждает состояние чужого стенда, наличие опубликованных Docker-образов или успешный запуск текущего checkout. Внутренние журналы развёртываний, идентификаторы ресурсов и реквизиты в публичный репозиторий не включены.

## Проверки из исходников

Подготовьте окружение по [руководству разработки](../../docs/rewrite/DEVELOPMENT.md). Из корня репозитория:

```bash
bash scripts/verify-rewrite.sh
python3 -m unittest discover -s deploy/rewrite/yc
```

Общий verifier проверяет backend, frontend, Playwright в закреплённом Linux image, Compose, CPU Docker images, shell-скрипты, Bats-тесты, release bundle и checksums. Требуются Docker, npm, backend-окружение, `sqlite3`, `shellcheck` и `bats`. Команды не создают облачные ресурсы. Unit-тесты облачной автоматизации используют имитацию облака.

Без флагов ML и CUDA явно пропускаются. Для десяти реальных весов и regression-проверки четырёх примеров:

```bash
SPINA_BIFIDA_MODELS_ROOT=/absolute/path/to/models \
  bash scripts/verify-rewrite.sh --with-models
```

Для NVIDIA-host используйте `--with-gpu`; CPU-проверка не подтверждает GPU. Canonical PNG/Grad-CAM и visual baselines проверяются в Linux Docker images, поэтому успешный host-тест не заменяет эти проверки.

## Проверка собственного стенда

После [установки](../../docs/rewrite/OPERATIONS.md) проверьте `/health/live`, `/health/ready`, миграции, healthy worker и свежий heartbeat каждой реплики после загрузки обеих версий моделей. Выполните анализ четырёх встроенных примеров на `v1` и `v2`, проверьте ROI/heatmap, обратную связь и восстановление backup.

Отдельно проверьте вход, запрет доступа к чужим анализам, CSRF и административные ограничения. При включённом Postbox проверьте подтверждение email, запрет входа до одобрения, модерацию, сброс и смену пароля. Принятие письма провайдером само по себе не подтверждает его доставку.

Для source-install каталог `SPINA_INSTALL_ROOT` должен указывать на `deploy/rewrite` внутри checkout. Для IP mode используется HTTP; рабочий контур с чувствительными данными требует настроенного HTTPS и контроля доступа. Веса, secrets, backups и runtime-данные всегда хранятся вне публичного Git.
