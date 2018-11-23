# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = CNNSurv
SOURCEDIR     = docs_source
BUILDDIR      = docs_build

EXCLUDE_PRIVATE= _source,_asdict,_fields,_field_defaults,_field_types,_replace,_make
APIDOC_OPTIONS='members,private-members,undoc-members,show-inheritance,inherited-members,change-1'

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

docs:
	rm -rf docs_source/api
	SPHINX_APIDOC_OPTIONS=${APIDOC_OPTIONS} sphinx-apidoc -M -o docs_source/api/ Sources Sources/experimental.py
	sed -i.bak 's/change-1:/exclude-members: $(EXCLUDE_PRIVATE)/' $(SOURCEDIR)/api/*.rst
	$(MAKE) html

clean:
	rm -rf docs_build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


