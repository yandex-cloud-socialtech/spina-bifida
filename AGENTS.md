# AGENTS.md

## Область действия и актуальная версия

Инструкции относятся ко всему репозиторию. Общаться по-русски, сохраняя имена файлов, команды и идентификаторы.

Активное приложение — Flask API + React/TypeScript + отдельный ML-worker на одной VM под Docker Compose. Данные и очередь заданий хранятся локально в SQLite (WAL), изображения — в файловом хранилище. Object Storage опционален; YDB новому приложению не нужен.

Публичный ML-контур ограничен YOLOv8 и DenseNet121/201. Публичное дерево содержит код инференса готовых моделей. Материалы обучения в этот выпуск не входят. Датасеты, веса и результаты запусков в Git не добавлять. Единственные исходные медицинские изображения в репозитории — четыре файла `apps/frontend/public/examples/`.

## Где искать код и документацию

| Задача | Основные пути |
|---|---|
| API, аутентификация, пользователи, анализы | `apps/backend/src/spina_bifida/`: `auth/`, `admin/`, `analyses/`, `api/` |
| ML и обработка очереди | `apps/backend/src/spina_bifida/inference/`, `apps/backend/src/spina_bifida/queue/` |
| Хранилища, retention, аудит и выгрузка | `apps/backend/src/spina_bifida/storage/`, `apps/backend/src/spina_bifida/observability/` |
| Схема БД и backend-тесты | `apps/backend/migrations/`, `apps/backend/tests/` |
| Интерфейс, API-клиент, переводы, стили | `apps/frontend/src/`: `features/`, `api/`, `i18n/`, `styles/` |
| Установка, Compose, Traefik, облачный стенд | `deploy/rewrite/`, `deploy/rewrite/yc/` |

Читать документацию по задаче, без обязательного обхода всего репозитория:

- [Разработка и окружение](docs/rewrite/DEVELOPMENT.md) — зависимости, локальные проверки, ML и visual acceptance.
- [ML-конвейер](docs/rewrite/MODEL-PIPELINE.md) — preprocessing, ROI, версии моделей и golden baseline.
- [Безопасность](docs/rewrite/SECURITY.md) — доступ, секреты, чувствительные данные.
- [Эксплуатация](docs/rewrite/OPERATIONS.md) — установка, конфигурация, пользователи, backup/restore и обновление.
- [Yandex Cloud](docs/rewrite/YANDEX-CLOUD.md) — создание стенда; [acceptance](deploy/rewrite/ACCEPTANCE.md) — команды проверки и ограничения.
- [Руководство пользователя](docs/rewrite/USER-GUIDE.md) — пользовательские сценарии.

Документация приложения — `docs/rewrite/`; общая информация о проекте и границах публикации — `docs/PROJECT.md`.

## Окружение и команды

Python **3.12**, Node.js **22.12–22.x**, npm **10–11**. Источники зависимостей: `apps/backend/pyproject.toml`, закреплённые `apps/backend/requirements/*.txt` и `apps/frontend/package-lock.json`. Для полной проверки нужны Docker Engine 24+ с Compose Plugin 2.24+, `sqlite3`, `shellcheck`, `bats`.

Каждый блок ниже запускается из корня репозитория.

Backend — подготовка и базовые проверки:

```bash
cd apps/backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/dev.txt
.venv/bin/pip install -r requirements/cpu.txt
.venv/bin/ruff check src tests
.venv/bin/mypy src/spina_bifida
.venv/bin/pytest -q -m 'not ml'
```

Для сфокусированной проверки передать `pytest` конкретный файл из `tests/unit/` или `tests/integration/`. Эти тесты используют временные каталоги и локальную SQLite; доступ к production/YDB и реальные веса для них не нужны.

Frontend — подготовка и базовые проверки:

```bash
cd apps/frontend
npm ci
npm run typecheck
npm test -- --run
npm run build
```

`npm run dev` запускает Vite. E2E: после `npx playwright install chromium` выполнить `npx playwright test` в `apps/frontend/`; по умолчанию используются mock API. Visual baseline проверять в закреплённом Linux Playwright image через общий verifier, не ослаблять screenshot-порог ради различий macOS/Linux.

Проверки установщика и облачной автоматизации:

```bash
bash deploy/rewrite/tests/test-compose.sh
bash deploy/rewrite/tests/test-health-domain.sh
shellcheck deploy/rewrite/install.sh deploy/rewrite/lib/*.sh deploy/rewrite/build-release.sh deploy/rewrite/tests/test-compose.sh scripts/verify-rewrite.sh
bats deploy/rewrite/tests/*.bats
python3 -m unittest discover -s deploy/rewrite/yc
```

Общая проверка приложения и release bundle:

```bash
bash scripts/verify-rewrite.sh
SPINA_BIFIDA_MODELS_ROOT=/absolute/path/to/models bash scripts/verify-rewrite.sh --with-models
```

`--with-models` проверяет реальные веса и golden-результаты в Linux `test-cpu` image. `--with-gpu` включает ML-проверки и требует NVIDIA-host. Без этих флагов ML/CUDA явно пропускаются; это не подтверждение работы моделей. Облачные unit-тесты запускаются отдельно от общего verifier.

Выбирать проверки по затронутой области; для изменения только документации достаточно перечитать diff, проверить пути/ссылки и выполнить `git diff --check`.

## Границы реализации

- API не загружает веса и не выполняет inference: этим занимается отдельный worker из того же backend-пакета. Очередь — `analyses/repository.py` и SQLite; не добавлять брокер, внешнее model API или plugin-архитектуру без отдельной архитектурной задачи.
- Backend: сохранять разделение routes → service → repository, использовать `extensions.db`, `storage/files.py`, `storage/object_storage.py` и существующий outbox. Изменения схемы оформлять Alembic-миграцией в `apps/backend/migrations/` и проверять migration-тестами.
- API-контракты менять согласованно с `apps/frontend/src/api/contracts.ts` и клиентом `apps/frontend/src/api/client.ts`. Сохранять same-origin, cookie-сессию, CSRF, проверку владельца задания/asset и формат ошибок `error: {code, params, request_id}`.
- Саморегистрация: `pending_email` → подтверждение почты → `pending` → решение администратора (`active`/`rejected`). Не допускать вход до `active` и не заменять модерацию разблокировкой. Смена/сброс пароля должны отзывать все сессии и прежние ссылки.
- UI: использовать Gravity UI, существующие компоненты, TanStack React Query и токены `apps/frontend/src/styles/`. Строки добавлять через i18next в `apps/frontend/src/i18n/{ru,en,kk,uz}.json`; основной язык и fallback — `ru`.
- Логирование и аудит — через `observability/logging.py` и `observability/audit.py`. Не записывать пароли, cookies, токены, изображения и ПДн пациентов в логи. Pending-файлы outbox не удалять до подтверждённой выгрузки; сохранять правила retention и квот.

## Конфигурация и эксплуатация

Настройки валидируются в `apps/backend/src/spina_bifida/config.py`; шаблон — [app.example.yaml](deploy/rewrite/config/app.example.yaml). YAML задаётся через `APP_CONFIG_FILE`; секрет сессии — `APP_SESSION_SECRET`, ключи Object Storage — `OBJECT_STORAGE_ACCESS_KEY` и `OBJECT_STORAGE_SECRET_KEY` из окружения. Postbox задаётся секцией `postbox` и отдельными `POSTBOX_ACCESS_KEY`/`POSTBOX_SECRET_KEY`; дополнительный защищённый env-файл подключается через `POSTBOX_ENV_FILE`. Не помещать секреты в YAML, исходники или команды с открытыми значениями.

Для ручного запуска настроить локальные каталоги данных по руководству разработки. Для установки из исходников использовать `deploy/rewrite/install.sh` с `SPINA_INSTALL_ROOT` по руководству эксплуатации; для создания облачного стенда — `deploy/rewrite/yc/deploy.py`. Deploy, выпуск версии и создание облачных ресурсов выполнять только в явно порученном scope.

`/health/ready` проверяет API, БД, схему и writable storage. Готовность ML подтверждают отдельно: healthy worker и свежий heartbeat каждой реплики после загрузки обеих версий моделей. Публичные порты публикует только Traefik (80/443).

## ML и медицинская семантика

- Обязательны обе версии `v1`/`v2`, по пять `.pt`-файлов; SHA-256 — в `deploy/rewrite/models/manifest.sha256`. Веса не входят в Git или release bundle и монтируются в worker read-only. Не вводить скрытый fallback с CUDA на CPU или на одну версию модели.
- Намеренные изменения preprocessing, exact ROI, порогов или Grad-CAM требуют клинического согласования и объяснённого обновления baseline: `apps/backend/tests/fixtures/expected_results.json`, `apps/backend/tests/ml/`. Display padding не должен менять вход классификаторов и числовые результаты.
- Пороги задаются в `inference.thresholds`; `warning_margin` управляет цветовой зоной UI, а не создаёт новую категорию результата.
- Score модели не является диагнозом или калиброванной клинической вероятностью. Сохранять эту семантику в интерфейсе, API и документации.

При изменении команд, структуры или границ приложения актуализировать этот файл; подробные процедуры хранить в профильной документации, не дублируя здесь глобальные правила агента.
