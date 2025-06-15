CXX := g++
INCLUDEDIR := include
EXTERNAL := external
MATHOPS := $(EXTERNAL)/mathops
LIBMATHOPS := $(MATHOPS)/build/libmathops.a
CXXFLAGS := -I$(INCLUDEDIR) -I$(MATHOPS)/include
LDFLAGS := -L$(MATHOPS)/build -lmathops
BUILDDIR := build
OBJDIR := $(BUILDDIR)
SRCDIR := src
SRCFILES := $(wildcard $(SRCDIR)/*.cpp)
OBJFILES := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCFILES))
TEST := $(BUILDDIR)/test
DEPS := $(OBJFILES:.o=.d)

.PHONY := all clean

all: $(OBJFILES) $(TEST) $(LIBMATHOPS)

$(LIBMATHOPS):
	make -C $(MATHOPS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) -MMD -MP -I $(INCLUDEDIR) -c $< -o $@

-include $(DEPS)

$(TEST): tests/test.cpp $(OBJFILES) $(LIBMATHOPS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ tests/test.cpp $(OBJFILES)

clean:
	rm -rf $(BUILDDIR)
	make -C $(MATHOPS) clean