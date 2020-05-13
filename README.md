# image_plotting_tools

For making images and movies from microscopy imagesDetails for the systems it currently works on

## Installation Instructions
### Mac:

OS X ver. 10.15.4 (macOS Catalina)
ffmpeg ver 4.2.2

***To install:***

1. Install homebrew by running the following command in a terminal window:

		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

1. Install ffmpeg by running:

		brew install ffmpeg

1. onfirm that it is installed correctly by typing ffmpeg at the prompt

### Windows:

Windows 10
ffmpeg ver 4.2.2 - from here: https://ffmpeg.zeranoe.com/builds/

***To install:***

1. Download ffmpeg from here: https://ffmpeg.zeranoe.com/builds/
1. Unzip the contents into a folder on your computer
1. Search for Advanced System Properties from the search menu
1. Click on the Environment Variables button at the bottom
1. In the top half of the window under User Variables, select Path and hit the Edit button
1. Add the path to the directory where you unzipped ffmpeg and add `\bin\` to it - e.g. `c:\ffmpeg\bin\`
1. Click OK
1. Check to see if the path to the ffmpeg folder was added to the path variable. There should now be a `;` before and after it
1. Confirm that it is installed correctly by typing ffmpeg at the prompt

## Example Usage

Import appropriate toolboxes
```python
import glob as glob
from image_plotting_tools import *
from interactive_plotting_tools import *
```

Overlay your images
```python
file_list = glob.glob('example1/*.tif')
interactive_overlay_images(file_list)
```

Make a movie
(where `example3/finale_test.tif'` is the multicolor image overlay created in the cell above)
```python
movie_maker_widget('example3/final_test.tif')

```
