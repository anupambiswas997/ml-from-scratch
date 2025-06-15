CXX := g++
INCLUDEDIR := include
BUILDDIR := build
OBJDIR := $(BUILDDIR)
SRCDIR := src
SRCFILES := $(wildcard $(SRCDIR)/*.cpp)
OBJFILES := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCFILES))
TEST := $(BUILDDIR)/test
DEPS := $(OBJFILES:.o=.d)

.PHONY := all clean

all: $(OBJFILES) $(TEST)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) -MMD -MP -I $(INCLUDEDIR) -c $< -o $@

$(TEST): tests/test.cpp $(OBJFILES)
	$(CXX) -I $(INCLUDEDIR) -o $@ $^

clean:
	rm -rf $(BUILDDIR)