.PHONY: help clean html

help:
	@echo "Sphinx documentation build targets:"
	@echo ""
	@echo "  make html       build HTML documentation"
	@echo "  make clean      remove build artifacts"
	@echo "  make serve      build HTML and serve on localhost:8000"
	@echo "  make livehtml    rebuild HTML on source changes (requires sphinx-autobuild)"
	@echo "  make latexpdf    build LaTeX/PDF documentation"

clean:
	rm -rf _build/

html:
	sphinx-build -W --keep-going -b html . _build/html

serve: html
	python -m http.server -d _build/html 8000

livehtml:
	sphinx-autobuild -b html . _build/html

latexpdf:
	sphinx-build -b latex . _build/latex
	cd _build/latex && make all-pdf
