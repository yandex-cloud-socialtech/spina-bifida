# Развёртывание нового приложения с нуля

Для автоматического создания VM и установки приложения см. [короткую инструкцию Yandex Cloud](YANDEX-CLOUD.md). Ниже — ручная установка и эксплуатация.

Руководство относится к Flask/React-приложению из `main`: `apps/backend`, `apps/frontend`, `deploy/rewrite`. YDB для этой установки не нужен.

На одной VM работают Traefik, frontend, API, worker и одноразовый сервис миграции. SQLite, изображения, feedback и логи хранятся в Docker volumes. Object Storage необязателен. Пользователи создаются локально администратором; автоматического переноса из старого приложения нет.

## 1. Подготовить сервер

Поддерживаются Ubuntu 22.04/24.04, архитектура x86_64, доступ по SSH и `sudo`. Нужны свободные TCP 80/443 и исходящий доступ к источникам Docker-образов, Debian, npm, PyPI и PyTorch. Для HTTPS заранее направьте DNS A/AAAA домена на VM и разрешите входящие 80/443. Не оставляйте AAAA, указывающую на другой сервер. В Yandex Cloud проверьте security groups именно сетевого интерфейса VM: стандартная группа может разрешать SSH, но не HTTP/HTTPS. Подключите согласованную группу с входящими TCP 80/443, сохранив SSH; не расширяйте общую группу других VM без необходимости.

Ранее CPU-приёмка выполнялась на 8 vCPU / 16 GiB RAM. Это проверенная конфигурация, а не минимальные требования. Текущие команды проверки и ограничения указаны в [документе проверки](../../deploy/rewrite/ACCEPTANCE.md). Для диска учитывайте образы, кеш сборки, модели, данные и backups; квоты приложения не резервируют место на диске.

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssl python3
```

Docker Engine (от 24) и Compose установщик проверит и при необходимости установит сам. Python/Node-зависимости приложения устанавливаются внутри образов.

## 2. Получить исходники

На сервере под обычным SSH-пользователем с доступом к репозиторию:

```bash
git clone --branch main --single-branch \
  https://github.com/yandex-cloud-socialtech/spina-bifida.git \
  "$HOME/spina-bifida-src"
cd "$HOME/spina-bifida-src"
git rev-parse HEAD
cd deploy/rewrite
export SPINA_INSTALL_ROOT="$PWD"
```

Если сервер не имеет доступа к Git, передайте полный `git archive` выбранного commit с рабочей машины и распакуйте его на сервере, например в `/opt/spina-bifida-src`. После этого перейдите в его `deploy/rewrite` и так же задайте `SPINA_INSTALL_ROOT="$PWD"`. Не переносите `.env`, пользовательские базы и ключи от другого стенда.

**Для сборки из исходников каталог установки должен оставаться внутри checkout.** Compose использует `../../apps/backend` и `../../apps/frontend`. Не запускайте source-install с каталогом по умолчанию `/opt/spina-bifida`: там этих исходников нет. Во всех последующих командах передавайте тот же `SPINA_INSTALL_ROOT`.

## 3. Разместить модели

Скачайте оба комплекта по [публичным HTTPS-ссылкам](../../deploy/rewrite/models/README.md). Из каталога `deploy/rewrite`:

```bash
bash ./download-models.sh "$HOME/spina-bifida-models"
sudo install -d -m 0750 /opt/spina-bifida-models
sudo cp -R "$HOME/spina-bifida-models/v1" "$HOME/spina-bifida-models/v2" /opt/spina-bifida-models/
sudo bash -c 'cd /opt/spina-bifida-models && sha256sum -c "$1"' \
  bash "$SPINA_INSTALL_ROOT/models/manifest.sha256"
```

Ожидаются десять `OK`. Скрипт скачивания проверяет SHA-256 до замены каждого файла. Для установки без сети выполните скачивание на другой машине и передайте каталоги `v1` и `v2` на сервер. Установщик повторяет проверку и назначает права `root:10001`, каталоги `0750`, файлы `0640`.

Каждая worker-реплика при запуске загружает **обе версии** до первого heartbeat. Отсутствующая модель останавливает worker; одного работающего API недостаточно для готовности приложения.

## 4. Настроить источники сборки при сетевых ограничениях

При доступных стандартных источниках этот шаг пропустите. Если Docker Hub, Debian или npm завершаются таймаутом, сначала проверьте доступность источников с VM. Не отключайте TLS и проверку подписей пакетов.

До первой установки создайте конфигурацию, если её ещё нет:

```bash
sudo install -m 0600 .env.example .env
sudoedit .env
```

Не перезаписывайте существующую `.env`: она содержит секрет сессий и настройки установленного стенда. Параметры, использованные на demo:

```dotenv
BUILD_PYTHON_IMAGE=cr.yandex/mirror/library/python:3.12-slim-bookworm
BUILD_DEBIAN_MIRROR=https://mirror.yandex.ru/debian
BUILD_DEBIAN_SECURITY_MIRROR=https://mirror.yandex.ru/debian-security
BUILD_DOCKERFILE_FRONTEND=dockerhub.timeweb.cloud/docker/dockerfile:1.7
BUILD_NODE_IMAGE=dockerhub.timeweb.cloud/library/node:22-bookworm-slim
BUILD_NGINX_IMAGE=dockerhub.timeweb.cloud/nginxinc/nginx-unprivileged:1.27-alpine
BUILD_NPM_REGISTRY=https://registry.npmmirror.com
TRAEFIK_IMAGE=dockerhub.timeweb.cloud/library/traefik:v3.3
```

Зеркало Яндекса предпочтительно там, где доступен нужный пакет/образ. На момент проверки оно предоставляло указанный Python-образ и Debian-пакеты, но не остальные требуемые Docker tags. Доступность зеркал меняется: проверяйте её перед установкой. npm сохраняет проверку `integrity` из `package-lock.json`, apt — подписи Debian. По умолчанию сборка продолжает использовать исходные upstream-адреса; настройки зеркал необязательны и не меняют сеть ОС.

### Если недоступен CDN CPU PyTorch

Для Python 3.12 / Linux x86_64 скачайте на машине с доступом к официальному CDN следующие wheels и проверьте суммы из официальных индексов [torch](https://download.pytorch.org/whl/cpu/torch/) и [torchvision](https://download.pytorch.org/whl/cpu/torchvision/):

```bash
mkdir -p cpu-wheels
cd cpu-wheels
curl -fL --retry 2 -o 'torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
curl -fL --retry 2 -o 'torchvision-0.22.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl' \
  'https://download-r2.pytorch.org/whl/cpu/torchvision-0.22.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
cat > SHA256SUMS <<'SUMS'
8f8b3cfc53010a4b4a3c7ecb88c212e9decc4f5eeb6af75c3c803937d2d60947  torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
b5fa7044bd82c6358e8229351c98070cf3a7bf4a6e89ea46352ae6c65745ef94  torchvision-0.22.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
SUMS
sha256sum -c SHA256SUMS  # на macOS: shasum -a 256 -c SHA256SUMS
```

Передайте wheels и `SHA256SUMS` в `<checkout>/apps/backend/requirements/wheels/`, повторите проверку сумм на сервере. Этот каталог исключён из Git. В `.env` добавьте обе строки:

```dotenv
BUILD_TORCH_CPU_INDEX_URL=
BUILD_PIP_FIND_LINKS=/build/requirements/wheels
```

Пустое значение в первой строке намеренное: оно отключает обращение к CPU CDN при сборке. Остальные Python-зависимости по-прежнему загружаются из PyPI. Это обход недоступности конкретного CDN, а не полностью offline-установка. После изменения версий torch/torchvision нужно заново выбрать совместимые wheels и проверить официальные суммы.

## 5. Выполнить CPU-установку

Запускайте из `deploy/rewrite` в интерактивной SSH-сессии. Замените домен и email своими значениями. Для ACME нужен действующий контактный email: Let’s Encrypt отклоняет `example.com`; адреса ниже — только placeholders:

```bash
sudo env SPINA_INSTALL_ROOT="$SPINA_INSTALL_ROOT" \
  MODELS_DIR=/opt/spina-bifida-models \
  APP_ACCESS_MODE=domain APP_DOMAIN=sb.example.org \
  ACME_EMAIL=ops@example.org ADMIN_EMAIL=admin@example.org \
  bash ./install.sh install --force-cpu
```

Для теста по IP замените настройки домена на `APP_ACCESS_MODE=ip`; адрес приложения будет `http://IP`. Такой режим передаёт логин и пароль по HTTP, поэтому для удалённой эксплуатации используйте HTTPS.

Installer создаёт `.env` при отсутствии, генерирует секрет сессий, создаёт `config/app.yaml`, проверяет модели, собирает образы, выполняет миграции SQLite и проверяет здоровье всех сервисов. Секреты и конфигурация получают права `0600`; `app.yaml` принадлежит UID/GID `10001` для чтения контейнером.

В конце вводится пароль первого администратора с подтверждением (минимум 12 символов) и язык (`ru` по умолчанию). Пароль не задаётся через `ADMIN_PASSWORD`. При неинтерактивном запуске создание администратора пропускается — выполните следующий шаг отдельно.

### Ручной сертификат через DNS-01

Если Let’s Encrypt не достигает VM для HTTP-01, сертификат можно выпустить на рабочей машине с Certbot через ручной TXT challenge. В этом режиме **Traefik обслуживает сертификат, но не выпускает и не продлевает его**. На demo выбран именно этот вариант; IAM-ключи DNS на VM не передаются.

На машине с установленным Certbot выполните (подставьте действующие домен и email):

```bash
certbot certonly --manual --preferred-challenges dns --agree-tos --no-eff-email \
  --email ops@example.org --domain sb.example.org \
  --config-dir "$HOME/.local/share/spina-certbot/config" \
  --work-dir "$HOME/.local/share/spina-certbot/work" \
  --logs-dir "$HOME/.local/share/spina-certbot/logs"
```

Certbot напечатает имя `_acme-challenge.sb.example.org` и одноразовое значение TXT. Добавьте его в DNS-панели домена (или через `yc dns zone add-records` при Yandex Cloud DNS), дождитесь видимости записи на авторитетных NS и только затем нажмите Enter. Если challenge-имя уже имеет CNAME, TXT размещается по адресу CNAME, а не рядом с ним. После успешной выдачи удалите только TXT этой проверки.

Передайте `fullchain.pem` и `privkey.pem` из напечатанного Certbot каталога по SSH на VM. Разместите их в `/etc/spina-bifida/tls/`: каталог `root:root 0700`, оба файла `root:root 0600`. При копировании учитывайте, что файлы Certbot в `live/` — символические ссылки: передавать нужно содержимое. В `.env` задайте:

```dotenv
TRAEFIK_TLS_MODE=external
TRAEFIK_CERTS_DIR=/etc/spina-bifida/tls
```

Пример безопасной передачи из локального каталога Certbot (путь подставьте из его вывода):

```bash
ssh user@server 'sudo install -d -m 0700 /etc/spina-bifida/tls'
cat /path/to/live/sb.example.org/fullchain.pem | ssh user@server \
  "sudo sh -c 'umask 077; cat > /etc/spina-bifida/tls/fullchain.pem'"
cat /path/to/live/sb.example.org/privkey.pem | ssh user@server \
  "sudo sh -c 'umask 077; cat > /etc/spina-bifida/tls/privkey.pem'"
```

Команды предполагают настроенный `sudo` без интерактивного запроса пароля; иначе перенесите файлы в закрытый промежуточный каталог и установите их через интерактивную SSH-сессию. Приватный ключ не выводится в терминал.

Затем выполните обычную команду установки из шага 5. Installer создаст маршруты без ACME resolver и подключит сертификат к Traefik; API, frontend и worker приватный ключ не получают. Режим по умолчанию остаётся `acme` с автоматическим HTTP-01.

**Продление вручную:** до истечения срока повторите команду Certbot, выполните новый TXT challenge, замените оба файла на VM с прежними правами и перезапустите Traefik:

```bash
sudo docker compose --env-file .env -f compose.yaml restart traefik
curl --fail https://sb.example.org/health/ready
openssl s_client -connect sb.example.org:443 -servername sb.example.org </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer -dates
```

`certbot renew` без authentication hook не выполняет ручной DNS challenge автоматически. Установите собственное напоминание о продлении. Каталог сертификата `/etc/spina-bifida/tls` не входит в backup приложения; сохраните его отдельно защищённо либо выпустите новый сертификат при восстановлении на другой VM. Сохраняйте каталог Certbot также: там находятся ACME account и ключи, доступ к ним должен быть ограничен.

## 6. Создать администратора и добавить пользователей

Если администратор не создан:

```bash
sudo docker compose --env-file .env -f compose.yaml \
  run --rm --no-deps api \
  flask --app spina_bifida:create_app create-admin --email admin@example.org
```

Команда скрыто запросит пароль, подтверждение и язык. Email должен быть уникальным; повторный запуск не меняет пароль существующей записи.

1. Откройте адрес приложения и войдите под администратором.
2. Прочитайте и примите условия использования; они подтверждаются при каждом новом входе.
3. Откройте `/admin`.
4. Введите email пользователя, пароль от 12 символов, роль `user` или `admin`, язык `ru`, `en`, `kk` или `uz`.
5. Нажмите «Добавить пользователя» и передайте реквизиты пользователю по согласованному защищённому каналу.
6. Проверьте вход созданным пользователем и запуск анализа на встроенных обезличенных примерах.

Администратор может менять роль и язык, блокировать, разблокировать и удалять записи. Обычный пользователь не имеет доступа к управлению пользователями. Последнего активного администратора нельзя удалить или заблокировать.

Пользователи могут зарегистрироваться на `/register`, подтвердить почту и дождаться модерации на `/admin`. До одобрения войти нельзя. В меню профиля доступны смена пароля и, для администратора, ссылка на админ-панель; на экране входа — восстановление пароля. Подпись «Временный пароль» у создаваемого администратором пользователя не означает обязательную смену при первом входе. Сохраняйте пароль администратора в менеджере паролей; не помещайте его в Git или логи.

## 7. Проверить установку

```bash
sudo env SPINA_INSTALL_ROOT="$SPINA_INSTALL_ROOT" bash ./install.sh status
sudo docker compose --env-file .env -f compose.yaml ps --all
curl --fail https://sb.example.org/health/ready
sudo docker stats --no-stream
```

Traefik/frontend/API/worker должны быть `healthy`, `migrate` — `Exited (0)`. Readiness API проверяет БД, схему и storage; модели подтверждаются healthy worker и реальным анализом.

В браузере проверьте: вход и условия, добавление пользователя, блокировку/разблокировку, загрузку примеров, завершение анализа отдельно на `v1` и `v2`, изображения ROI/Heatmap и feedback. После завершения очередь должна быть пустой. Не считайте установку принятой только по HTTP 200.

## 8. Настроить работу и применить изменения

`config/app.yaml` содержит версию моделей, число worker (`1..4`), таймаут, пороги, хранение данных, ротацию логов, язык и Object Storage. По умолчанию: `v2`, один worker, таймаут 300 секунд, `feedback_only`, временные изображения 24 часа, логи 30 дней, Object Storage выключен.

Пороги `plane_detection`, `quality`, `assessment` и `warning_margin` находятся в `inference.thresholds`. Последний параметр задаёт ширину жёлтой зоны интерфейса вокруг порога и не меняет текстовую классификацию результата.

```bash
sudoedit config/app.yaml
sudo docker compose --env-file .env -f compose.yaml \
  up -d --no-deps --force-recreate --scale worker=1 api worker
```

Если меняете `max_parallel_jobs`, укажите такое же число в `--scale worker=N`. Каждая реплика держит обе версии моделей в памяти. Увеличивайте параллельность только после измерения RAM/VRAM.

Пересоздание контейнеров применяется и после изменения `.env`; простой `restart` не обновляет env и может сохранить старый bind mount при замене файла редактором. Права `config/app.yaml` должны оставаться `0600`, владелец `10001:10001`. Для смены IP/domain используйте installer с явными `APP_ACCESS_MODE`, `APP_DOMAIN`, `ACME_EMAIL` и `MODELS_DIR`: одного изменения YAML недостаточно для маршрутов Traefik.

Object Storage: включите `object_storage.enabled`, задайте endpoint, bucket и нужные виды выгрузки в YAML; `OBJECT_STORAGE_ACCESS_KEY` и `OBJECT_STORAGE_SECRET_KEY` — только в `.env`. При недоступном storage outbox остаётся локально; контролируйте свободное место.

## 9. Резервное копирование и восстановление

```bash
sudo env SPINA_INSTALL_ROOT="$SPINA_INSTALL_ROOT" bash ./install.sh backup
# Подставьте имя архива, напечатанное командой:
sudo bash -c 'cd /var/backups/spina-bifida && sha256sum -c backup-YYYYMMDDTHHMMSSZ.tar.gz.sha256'
```

Архивы по умолчанию находятся на хосте в `/var/backups/spina-bifida` (`SPINA_BACKUP_ROOT`), а не в пути `backup.destination` внутри контейнера. Backup временно останавливает worker, сохраняет согласованную SQLite-копию, feedback, runtime-конфигурацию и секреты, затем возвращает worker. Retention — `backup.retention_count` (по умолчанию 7). Автоматическое расписание не устанавливается: организуйте его отдельно и копируйте архивы за пределы VM.

Backup не содержит веса, Docker-образы и полный временный каталог jobs. Храните модели и соответствующие образы отдельно. Архив содержит секреты, поэтому доступ к нему ограничен правами `0600`.

```bash
sudo env SPINA_INSTALL_ROOT="$SPINA_INSTALL_ROOT" bash ./install.sh restore \
  /var/backups/spina-bifida/backup-YYYYMMDDTHHMMSSZ.tar.gz
```

Restore проверяет суммы, создаёт emergency snapshot, восстанавливает данные и runtime, затем проверяет сервисы. Используйте его в согласованное окно: текущие данные заменяются выбранной копией.

## 10. Обновление и перезапуск

Перед обновлением сделайте backup и сохраните исходный commit и image tags. Доставьте проверенную новую версию исходников в тот же checkout, сохранив `.env`, `config/app.yaml` и отдельно расположенные модели. Используйте уникальную версию образов:

```bash
sudo env SPINA_INSTALL_ROOT="$SPINA_INSTALL_ROOT" \
  bash ./install.sh update --version 0.1.1 --force-cpu
```

Installer создаёт ещё один pre-update backup. После обновления повторите health и реальный анализ. Не удаляйте старые images до окончания проверки восстановления.

Перезапуск без загрузки образов (пример для одного CPU-worker):

```bash
sudo docker compose --env-file .env -f compose.yaml stop
sudo docker compose --env-file .env -f compose.yaml \
  up -d --pull never --scale worker=1
```

Это подтверждает использование локальных образов, но не доказывает работу при физически заблокированной исходящей сети. Не выполняйте `docker compose down -v`: это удаляет данные приложения.

## GPU и готовые release bundles

GPU: уберите `--force-cpu`. Установщик обнаруживает NVIDIA, при необходимости устанавливает драйвер и просит перезагрузку; после reboot повторите установку. Требуются рабочие `nvidia-smi` и NVIDIA Container Toolkit. Для ручных Compose-команд добавляйте `-f compose.gpu.yaml` после `-f compose.yaml`, иначе worker получит CPU-конфигурацию. GPU-приёмка выполняется отдельно, CPU-проверка её не заменяет.

Release bundle — альтернативный путь, когда образы уже опубликованы в доступном registry. `deploy/rewrite/build-release.sh VERSION` требует `RELEASE_BACKEND_CPU_IMAGE`, `RELEASE_BACKEND_GPU_IMAGE`, `RELEASE_FRONTEND_IMAGE` с реальными immutable `@sha256` references. Архив включает installer/config/Compose и checksum manifests, но не исходники, веса или образы. Проверьте внешний `.sha256`, распакуйте архив и запустите его `install.sh install`: для `APP_IMAGE_MODE=pull` допустим стандартный `/opt/spina-bifida`. Наличие публичного комплекта готовых образов этим руководством не предполагается.

## Диагностика

```bash
sudo docker compose --env-file .env -f compose.yaml logs --tail=100 api worker traefik
sudo env SPINA_INSTALL_ROOT="$SPINA_INSTALL_ROOT" bash ./install.sh status
df -h /
sudo docker stats --no-stream
```

- Таймаут registry/CDN: проверьте DNS и исходящий доступ с VM и из сборки; настройте доступные зеркала по шагу 4.
- Model checksum mismatch: получите правильный файл; не отключайте проверку.
- Worker unhealthy: проверьте обе версии моделей, права и память; `/health/ready` не заменяет эту проверку.
- HTTP 507: освободите место или скорректируйте retention; по умолчанию приложение оставляет 2 GiB свободными.
- Нет HTTPS: проверьте DNS, 80/443, ACME email и журнал Traefik; не принимайте самоподписанный сертификат как успешную установку.

## Yandex Cloud Postbox

Для регистрации и сброса пароля настройте подтверждённый домен в Postbox и добавьте выданные DKIM CNAME в авторитетную DNS-зону. Используйте отдельный сервисный аккаунт того же каталога с ролью `postbox.sender` и ограничьте отправителя в адресе Postbox. Подробности — [официальное руководство](https://yandex.cloud/ru/docs/tutorials/serverless/send-emails-aws-sdk-python).

В `config/app.yaml`:

```yaml
postbox:
  enabled: true
  endpoint: https://postbox.cloud.yandex.net
  region: ru-central1
  sender: noreply@sb.example.org
  public_url: https://sb.example.org
```

Создайте на VM `/etc/spina-bifida/postbox.env` (`root:root`, `0600`) с переменными `POSTBOX_ACCESS_KEY` и `POSTBOX_SECRET_KEY`. Значения вводите защищённым способом; не помещайте их в команды, Git или YAML. В существующую `.env` добавьте только путь `POSTBOX_ENV_FILE=/etc/spina-bifida/postbox.env`. Compose подключает этот файл дополнительно к основной `.env`; требуется Compose 2.24+ (optional env file). Для локальных тестов можно использовать игнорируемый `.postbox.env`.

После миграции и изменения конфигурации пересоздайте API/worker. При `postbox.enabled: false` создание пользователей администратором, вход и смена пароля из профиля доступны, публичная регистрация и запрос сброса возвращают `503 mail_unavailable`. Ошибка отправки при модерации сохраняет заявку в ожидании, чтобы администратор мог повторить действие.

Проверьте регистрацию на доступный вам email, запрет входа до подтверждения и одобрения, письмо о решении, вход, восстановление и смену пароля. Не считайте ответ Postbox об отправке доказательством доставки во входящие. Ключи Postbox, как внешние TLS-сертификаты, не входят в стандартный backup приложения: сохраняйте этот env-файл отдельно в защищённом хранилище. Старые пользователи после миграции сохраняют доступ.
