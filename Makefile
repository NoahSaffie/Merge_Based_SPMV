CC = g++
CFLAGS = -g -Wall -Wextra -pedantic
OBJECTS = main.o mmio.o

driver: $(OBJECTS)
	$(CC) $(CFLAGS) -o driver -fopenmp $(OBJECTS)
main.o: main.cpp
	$(CC) $(CFLAGS) -fopenmp -c main.cpp -o main.o
mmio.o: mmio.c mmio.h
	gcc -g -Wall --std=c99 -c mmio.c -o mmio.o
clean:
	rm driver $(OBJECTS)
