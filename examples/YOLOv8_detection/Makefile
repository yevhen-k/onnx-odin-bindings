.PHONY: build clean run

clean:
	rm -f example

SHELL := bash
.ONESHELL:
build:
	odin build . -extra-linker-flags:"-v -Wl,-rpath=/thirdparty/onnxruntime/lib/,`pkg-config --libs opencv` -lstdc++" -out:./example

run: build
	./example
