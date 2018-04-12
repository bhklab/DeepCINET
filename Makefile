# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = CNNSurv
SOURCEDIR     = docs_source
BUILDDIR      = docs_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

docs: clean
	rm -rf docs_source/api
	SPHINX_APIDOC_OPTIONS='members,special-members,private-members,undoc-members,show-inheritance' sphinx-apidoc -o docs_source/api/ Sources
	$(MAKE) html

clean:
	rm -rf docs_build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


