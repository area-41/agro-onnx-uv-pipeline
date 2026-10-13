# ==============================================================================
# STAGE 1: Builder (Instalação e Build do Pacote com uv_build)
# ==============================================================================
FROM python:3.12-slim-bookworm AS builder

# Copia os executáveis oficiais do uv para a imagem
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Otimizações de ambiente para execução do uv em container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1. Instalação de dependências brutas com cache do uv (sem copiar todo o código fonte)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. Copia o código-fonte da aplicação
COPY . /app

# 3. Executa o build/instalação do pacote usando o backend uv_build (sem o grupo dev)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==============================================================================
# STAGE 2: Runner (Imagem final de produção ultraleve)
# ==============================================================================
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Copia o ambiente virtual montado (.venv) e a estrutura da aplicação
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Usuário não-root para execução segura no Edge / Cloud
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Executa o entrypoint CLI definido em [project.scripts] no pyproject.toml
CMD ["agro-onnx-uv-pipeline"]