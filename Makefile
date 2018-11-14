CC := gcc
CFLAGS := -g -Wall -O3 -std=gnu11 -pedantic
LDFLAGS := -lm
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	LDFLAGS += -lrt
endif


all: complete incomplete

complete: complete.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

incomplete: incomplete.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf *.dSYM complete incomplete
