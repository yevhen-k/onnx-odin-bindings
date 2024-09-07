.PHONY: clean object static shared

clean:
	rm -f libopencvbinding.so opencvbinding.o opencvbinding.a

object:
	gcc -c -Wall -Werror -fpic -I/usr/include/opencv4 -I/usr/include/opencv4/opencv2 `pkg-config --libs opencv` -lstdc++ opencvbinding.cpp

static: object
	ar rcs opencvbinding.a opencvbinding.o

shared: object
	gcc -shared -o libopencvbinding.so opencvbinding.o