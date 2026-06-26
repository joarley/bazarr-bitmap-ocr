#!/bin/sh
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Create group if it doesn't exist
if ! getent group "$PGID" > /dev/null 2>&1; then
    addgroup --gid "$PGID" appgroup
fi

# Create user if it doesn't exist
if ! getent passwd "$PUID" > /dev/null 2>&1; then
    adduser --disabled-password --no-create-home --gecos "" \
            --uid "$PUID" --ingroup "$(getent group "$PGID" | cut -d: -f1)" \
            appuser
fi

exec gosu "$PUID:$PGID" ocr-service
