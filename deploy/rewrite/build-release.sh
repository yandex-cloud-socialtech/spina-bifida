#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(unset CDPATH; cd -- "$(dirname -- "$0")" && pwd)
VERSION=${1:-}
[[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)*$ ]] || {
  echo "Usage: $0 <version>" >&2
  exit 2
}

BACKEND_CPU_REF=${RELEASE_BACKEND_CPU_IMAGE:-}
BACKEND_GPU_REF=${RELEASE_BACKEND_GPU_IMAGE:-}
FRONTEND_REF=${RELEASE_FRONTEND_IMAGE:-}
for reference in "$BACKEND_CPU_REF" "$BACKEND_GPU_REF" "$FRONTEND_REF"; do
  [[ $reference =~ ^[A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}$ ]] || {
    echo "Set RELEASE_BACKEND_CPU_IMAGE, RELEASE_BACKEND_GPU_IMAGE and RELEASE_FRONTEND_IMAGE to immutable @sha256 references" >&2
    exit 2
  }
done

OUTPUT_DIR=${RELEASE_OUTPUT_DIR:-$ROOT_DIR/dist}
STAGING=$(mktemp -d "${TMPDIR:-/tmp}/spina-release.XXXXXX")
CHECKSUMS=$(mktemp "${TMPDIR:-/tmp}/spina-release-checksums.XXXXXX")
PACKAGE="spina-bifida-$VERSION"
trap 'rm -rf "$STAGING"; rm -f "$CHECKSUMS"' EXIT
mkdir -p "$OUTPUT_DIR" "$STAGING/$PACKAGE/config" "$STAGING/$PACKAGE/models"

cp "$ROOT_DIR/install.sh" "$ROOT_DIR/download-models.sh" "$ROOT_DIR/compose.yaml" "$ROOT_DIR/compose.gpu.yaml" \
  "$ROOT_DIR/.env.example" "$ROOT_DIR/release-manifest.yaml" "$STAGING/$PACKAGE/"
cp -R "$ROOT_DIR/lib" "$ROOT_DIR/traefik" "$STAGING/$PACKAGE/"
cp "$ROOT_DIR/config/app.example.yaml" "$STAGING/$PACKAGE/config/"
cp "$ROOT_DIR/models/manifest.sha256" "$ROOT_DIR/models/source-url.txt" "$STAGING/$PACKAGE/models/"

sed -i.bak "s/^application_version: .*/application_version: $VERSION/" \
  "$STAGING/$PACKAGE/release-manifest.yaml"
rm "$STAGING/$PACKAGE/release-manifest.yaml.bak"
awk -v cpu="$BACKEND_CPU_REF" -v gpu="$BACKEND_GPU_REF" -v frontend="$FRONTEND_REF" '
  /^  backend_cpu:/ { print "  backend_cpu: " cpu; next }
  /^  backend_gpu:/ { print "  backend_gpu: " gpu; next }
  /^  frontend:/ { print "  frontend: " frontend; next }
  { print }
' "$STAGING/$PACKAGE/release-manifest.yaml" >"$STAGING/$PACKAGE/release-manifest.yaml.tmp"
mv "$STAGING/$PACKAGE/release-manifest.yaml.tmp" "$STAGING/$PACKAGE/release-manifest.yaml"
sed -i.bak \
  -e 's/^APP_IMAGE_MODE=.*/APP_IMAGE_MODE=pull/' \
  -e "s|^BACKEND_CPU_IMAGE=.*|BACKEND_CPU_IMAGE=$BACKEND_CPU_REF|" \
  -e "s|^BACKEND_GPU_IMAGE=.*|BACKEND_GPU_IMAGE=$BACKEND_GPU_REF|" \
  -e "s|^FRONTEND_IMAGE=.*|FRONTEND_IMAGE=$FRONTEND_REF|" \
  "$STAGING/$PACKAGE/.env.example"
rm "$STAGING/$PACKAGE/.env.example.bak"
(
  cd "$STAGING/$PACKAGE"
  find . -type f ! -name release-checksums.sha256 -print | LC_ALL=C sort | while IFS= read -r file; do
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "$file"
    else
      shasum -a 256 "$file"
    fi
  done >"$CHECKSUMS"
)
mv "$CHECKSUMS" "$STAGING/$PACKAGE/release-checksums.sha256"

ARCHIVE="$OUTPUT_DIR/$PACKAGE.tar.gz"
tar -C "$STAGING" -czf "$ARCHIVE" "$PACKAGE"
if command -v sha256sum >/dev/null 2>&1; then
  printf '%s  %s\n' "$(sha256sum "$ARCHIVE" | awk '{print $1}')" "$(basename "$ARCHIVE")" >"$ARCHIVE.sha256"
else
  printf '%s  %s\n' "$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')" "$(basename "$ARCHIVE")" >"$ARCHIVE.sha256"
fi
printf '%s\n' "$ARCHIVE"
