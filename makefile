CC = mpicc
CFLAGS = -Wall
TARGET = traffic
DEPS = Sweiss_Utilities.h
OBJ = traffic_circle.o Sweiss_Utilities.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf *.o $(TARGET)