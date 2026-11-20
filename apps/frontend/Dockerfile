# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:22-bookworm-slim
ARG NGINX_IMAGE=nginxinc/nginx-unprivileged:1.27-alpine

FROM ${NODE_IMAGE} AS dependencies
ENV CI=1
WORKDIR /app
COPY package.json package-lock.json ./
ARG NPM_REGISTRY=https://registry.npmjs.org
RUN npm ci --ignore-scripts --registry="$NPM_REGISTRY"

FROM dependencies AS validation
COPY . .
RUN npm run typecheck \
    && npm test -- --run \
    && npm run build

FROM node:22.12.0-bookworm-slim AS e2e-node

FROM mcr.microsoft.com/playwright:v1.63.0-noble AS test-e2e
COPY --from=e2e-node /usr/local/ /usr/local/
ENV CI=1
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --engine-strict
COPY . .
CMD ["npx", "playwright", "test"]

FROM ${NGINX_IMAGE} AS runtime
ARG APP_VERSION=0.1.0
LABEL org.opencontainers.image.title="Spina Bifida frontend" \
      org.opencontainers.image.version="$APP_VERSION"
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=validation /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
