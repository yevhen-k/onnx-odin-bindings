.PHONY: build run clean
SHELL := bash

clean:
	rm -f explore

.ONESHELL:
build:
	cd ..
	odin build explore_model_properties/ -extra-linker-flags:"-v -Wl,-rpath=/thirdparty/onnxruntime/lib/" -out:explore_model_properties/explore

run: build
	./explore