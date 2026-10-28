# Maintainer: zhaowenbo
CONTIKI_PROJECT = sensor-node gateway-node relay-node
all: $(CONTIKI_PROJECT)

# Contiki-NG path: hardcode host path (Cooja does not pass $HOME to make).
# Override with: make CONTIKI=/your/path/contiki-ng
CONTIKI ?= ${AUTHOR_HOME}/contiki-ng

# Include PRNG-371 source
PROJECT_SOURCEFILES += prng371.c

# Disable RPL routing (we use direct link-local comms)
MAKE_ROUTING = MAKE_ROUTING_NULLROUTING

include $(CONTIKI)/Makefile.include
