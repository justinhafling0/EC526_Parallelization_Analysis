.SUFFIXES:
.SUFFIXES: .o .cpp
#============================================================
TARGET1 = Heatmap_MPI

#pgc++ -std=c++11 -Minfo=accel -acc -ta=tesla integral_scaling.cpp gaussElim.cpp gaussQuadWeight.cpp -o integrate
TARGET2 = Heatmap_Serial
C_OBJS2 = Heatmap_Serial.o



C_SOURCES =

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

DATA_FILES =
#x8err.pdf  cosPIxerr.pdf x2p1inverr.pdf

CCX = g++
CXXFLAGS = -g -Wall
MPI = mpicxx
MPIFLAGS = -std=c++11 -L/share/pkg.7/openmpi/3.1.1/install/lib/
#-std=c99

#============================================================
all: $(TARGET1) $(TARGET2)

$(TARGET1) : Heatmap_MPI.cpp
	$(MPI) $(MPIFLAGS) $^ $(LIBDIRS) $(C_SOURCES) -o $@

$(TARGET2) :   $(C_OBJS2)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS) $(C_SOURCES) -o $@

.o:.cpp	$(MY_INCLUDES)
	$(CCX)  -c  $(CXXFLAGS) $<





# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites
#============================================================


clean:
	rm -f $(TARGET1) *~ *.o

tar: $(ALL_SOURCES) $(DATA_FILES)
	tar cvf HW3_code.tar $(ALL_SOURCES)  $(DATA_FILES)
