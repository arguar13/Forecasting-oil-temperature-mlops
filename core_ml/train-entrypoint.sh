#!/bin/sh
# Ver core_ml/train.Dockerfile: "poetry install" no funciona en este entorno
# para el lockfile de core_ml -- se exporta el lockfile a requirements.txt
# (determinista, versiones pineadas) y se instala con pip, no con el
# instalador de poetry. Corre en cada arranque del contenedor porque el
# lockfile llega por bind mount (docker-compose.yml::trainer.volumes), no
# horneado en la imagen; el volumen de cache de pip hace que solo la primera
# corrida sea lenta.
set -e
poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt --only main
pip install --no-cache-dir -q -r /tmp/requirements.txt
exec "$@"
