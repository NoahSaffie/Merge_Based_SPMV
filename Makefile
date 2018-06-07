CC = g++
CFLAGS = -g -Wall -Wextra -pedantic
OBJECTS = main.o mmio.o asm.o

driver: $(OBJECTS)
	$(CC) $(CFLAGS) -o driver -fopenmp $(OBJECTS)
main.o: main.cpp
	$(CC) $(CFLAGS) -march=native -ffast-math -fopenmp -c main.cpp -o main.o
mmio.o: mmio.c mmio.h
	gcc -g -Wall --std=c99 -c mmio.c -o mmio.o
asm.o: asm.S
	gcc -c asm.S -o asm.o
clean:
	rm driver $(OBJECTS)
