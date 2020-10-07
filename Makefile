all: build/my_program
	bash -c "display <(build/my_program)"

build/my_program: src/main.cu src/*.cuh
	mkdir -p build
	nvcc src/main.cu -o build/my_program

clean:
	rm -r build
