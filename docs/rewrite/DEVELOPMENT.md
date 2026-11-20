# Разработка Flask/React-приложения

## Поддерживаемое окружение

- Python 3.12;
- Node.js 22.12–22.x и npm 10–11;
- Docker Engine 24+ с Compose Plugin;
- Chromium для Playwright;
- `sqlite3`, `shellcheck` и `bats` для deploy-тестов.

Исходники приложения находятся в `apps/backend/` и `apps/frontend/`. ML-конвейер использует YOLOv8 и DenseNet121/201; исходники обучения этих моделей в текущем комплекте отсутствуют.

## Backend

```bash
cd apps/backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/dev.txt
.venv/bin/pip install -r requirements/cpu.txt
.venv/bin/ruff check src tests
.venv/bin/mypy src/spina_bifida
.venv/bin/pytest -q -m 'not ml'
```

Веса не входят в Git. Проверка реальных весов запускается из корня репозитория:

```bash
SPINA_BIFIDA_MODELS_ROOT=/absolute/path/to/models \
  bash scripts/verify-rewrite.sh --with-models
```

Точные PNG SHA-256 зависят от platform image codec, а Grad-CAM может иметь малые CPU backend differences. Поэтому verifier выполняет canonical golden acceptance в Linux `test-cpu` image, а не в host-Python.

Конфигурация backend берётся из YAML по `APP_CONFIG_FILE`, секрет сессии — только из `APP_SESSION_SECRET`. Для ручного запуска скопируйте `deploy/rewrite/config/app.example.yaml` и задайте абсолютные локальные каталоги данных.

## Frontend

```bash
cd apps/frontend
npm ci
npm run typecheck
npm test -- --run
npm run build
npx playwright install chromium
npx playwright test
```

Запуск на host удобен для диагностики. Canonical visual acceptance в `scripts/verify-rewrite.sh` работает в pinned Linux Playwright image: порог screenshot остаётся строгим, а разница macOS/Linux font rendering не делает verifier невоспроизводимым.

Обычные Playwright-тесты используют Vite и mock API. Live-проверка Compose включается переменными:

```bash
LIVE_E2E=1 \
PLAYWRIGHT_BASE_URL=http://127.0.0.1 \
E2E_ADMIN_EMAIL=admin@example.org \
E2E_ADMIN_PASSWORD='read-from-secure-input' \
npx playwright test e2e/admin.spec.ts
```

Пароль не сохраняйте в shell history: на стенде предпочтительно передать его через временно экспортированную переменную в закрытой сессии.

## Compose и полная проверка

```bash
cd deploy/rewrite
APP_ENV_FILE=.env.example APP_CONFIG_PATH=./config/app.example.yaml bash tests/test-compose.sh
shellcheck install.sh lib/*.sh build-release.sh tests/test-compose.sh
bats tests/*.bats
```

Единая проверка ветки:

```bash
bash scripts/verify-rewrite.sh
SPINA_BIFIDA_MODELS_ROOT=/absolute/path/to/models bash scripts/verify-rewrite.sh --with-models
```

`--with-gpu` допустим только на NVIDIA-host и автоматически включает `--with-models`. Скрипт без hardware-флагов явно сообщает о пропусках ML/CUDA, но все базовые проверки остаются обязательными.

## Правила изменения ML-контура

Контракт фиксируют `tests/fixtures/expected_results.json`, десять SHA-256 из `deploy/rewrite/models/manifest.sha256` и тесты `tests/ml/`. Любая намеренная смена preprocessing, ROI, порога или Grad-CAM требует отдельного клинического согласования и обновления baseline с объяснением.
