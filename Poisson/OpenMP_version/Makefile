SRCS	= main.c datatools.c utils.c jacobi.c gs.c
OBJS	= $(SRCS:.c=.o)
TARGET	= poisson.$(CC)

CC	= suncc
INCS	= 
DBG	= -g
OPT	= -fast -O3
PARA	= -xopenmp -xloopinfo
CFLAGS	= $(DBG) $(OPT) $(PARA) $(XOPTS)

LIBS	= -lm 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	@/bin/rm -f $(OBJS) *~ core*

realclean: clean
	@/bin/rm -f $(TARGET)


