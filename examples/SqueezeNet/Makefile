.PHONY: build clean run

clean:
	rm -f example

SHELL := bash
.ONESHELL:
build:
	cd ..
	odin build SqueezeNet/ -extra-linker-flags:"-v -Wl,-rpath=/thirdparty/onnxruntime/lib/" -out:SqueezeNet/example

run: build
	./example
