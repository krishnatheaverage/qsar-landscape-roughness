# Build the manuscript and supporting information.
# Needs a LaTeX toolchain (TeX Live / MiKTeX) for `make`, or `tectonic` for `make tectonic`.

all: main.pdf supporting_information.pdf

main.pdf: main.tex
	latexmk -pdf main.tex

# si_tables.tex is committed, but regenerate it from the results with:
#   python ../src/make_si_tables.py
supporting_information.pdf: supporting_information.tex si_tables.tex
	latexmk -pdf supporting_information.tex

# self-contained engine, no latexmk / TeX install needed
tectonic:
	tectonic main.tex
	tectonic supporting_information.tex

clean:
	latexmk -C 2>/dev/null || rm -f *.aux *.log *.out *.bbl *.blg *.toc *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all tectonic clean
