# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = CNNSurv
SOURCEDIR     = docs_source
BUILDDIR      = docs_build

EXCLUDE_PRIVATE= "_source,_asdict,_fields,_field_defaults,_field_types,_replace,_make"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

docs: clean
	rm -rf docs_source/api
	SPHINX_APIDOC_OPTIONS='members,private-members,undoc-members,show-inheritance,temp-to-change' sphinx-apidoc -o docs_source/api/ Sources
	sed -i .bak "s/temp-to-change:/exclude-members: $(EXCLUDE_PRIVATE)/" $(wildcard $(SOURCEDIR)/api/*.rst)
	$(MAKE) html

clean:
	rm -rf docs_build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


