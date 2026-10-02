#!/bin/sh
# Give the user from `docker run -u $(id -u):$(id -g)` a name and home directory, since
# their ID does not exist in the image and some tools fail when looking up the user.
if ! whoami > /dev/null 2>&1; then
  echo "aware:x:$(id -u):$(id -g):aware:/tmp:/bin/bash" >> /etc/passwd
fi
exec "$@"
