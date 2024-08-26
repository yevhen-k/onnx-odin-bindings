.PHONY: build run clean
SHELL := bash

clean:
	rm -f explore

.ONESHELL:
build:
	cd ..
	odin build explore_env/ -extra-linker-flags:"-v -Wl,-rpath=/thirdparty/onnxruntime/lib/" -out:explore_env/explore

run: build
	./explore