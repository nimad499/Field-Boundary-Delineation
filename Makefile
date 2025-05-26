.PHONY: all clean build postbuild

PYINSTALLER=pyinstaller
SCRIPT=src/gui.py
DISTDIR=executable
WORKDIR=$(DISTDIR)/build
SPECDIR=$(DISTDIR)/spec

PROJECT_ROOT:=$(shell pwd)
ENV_PATH:=$(PROJECT_ROOT)/.env/lib/python3.13/site-packages
GDAL_DATA_PATH:=$(ENV_PATH)/rasterio/gdal_data

COMMON_FLAGS=\
  --windowed \
  --paths "$(ENV_PATH)" \
  --hidden-import PIL._tkinter_finder \
  --additional-hooks-dir=hook \
  --add-data "$(GDAL_DATA_PATH):gdal_data" \
  --distpath $(DISTDIR) \
  --workpath $(WORKDIR) \
  --specpath $(SPECDIR)

all: build

build:
	$(PYINSTALLER) $(COMMON_FLAGS) $(SCRIPT)
	$(MAKE) postbuild

postbuild:
	@echo "Copying icon/ folder to $(DISTDIR)/gui/"
	mkdir -p $(DISTDIR)/gui/icon
	cp -r icon/* $(DISTDIR)/gui/icon/

clean:
	rm -rf $(DISTDIR)
