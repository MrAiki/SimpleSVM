GCC    = gcc
CFLAGS = -Wall -O0 -g

OBJS = test_simple_svm.o simple_svm.o

.SUFFIXES: .c .o

TARGET = test_simple_svm

all : $(TARGET)

clean :
	rm -f $(OBJS) $(TARGET)

$(TARGET) : $(OBJS)

.c.o : $<
	$(GCC) -c $(CFLAGS) $<

test_simple_svm : $(OBJS) test_simple_svm.o
	$(GCC) $^ -o $@ $(CFLAGS)

