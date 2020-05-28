# image_plotting_tools

For making images and movies from microscopy images in a Jupyter notebook. Requires the following packages:

`numpy`
`skimage`
`matplotlib`
`urlib`
`os`
`shutil`
`subprocess`
`ipywidgets`
`glob`
`csv`

***optional***
`czifile` - for reading in .czi files. Can be installed by running `pip install czifile` from the conda prompt (windows) or terminal (mac)

You should have all these packages installed by default if you installed python via Anaconda. If you get errors importing any of them try running from the conda prompt (Windows) or terminal (Mac) `conda install <package_name>`. If that doesn't work try installing the package via `pip install <package_name>`.

Also requires `ffmpeg` which runs from the command prompt. 

## FFMPEG Installation Instructions
### Mac:

OS X ver. 10.15.4 (macOS Catalina)
ffmpeg ver 4.2.2

***To install:***

1. Install homebrew by running the following command in a terminal window:

		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

1. Install ffmpeg by running:

		brew install ffmpeg

1. Confirm that it is installed correctly by typing ffmpeg at the prompt. You should see details about version number and installed codecs, etc. 

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
1. Confirm that it is installed correctly by typing ffmpeg at the prompt. You should see details about version number and installed codecs, etc. 

## Adding the files to your python path

If you want to be able to import these files in any notebook you should add their file location to your python path. 

### Mac:
1. Download the files from Github to a folder that will store your python code
1. Navigate to your home folder (look for the house in the list of favorites in a finder window)
1. Show hidden files by hitting `shift` + `command` + `.`
1. Right click on `.bash_profile` and select "Open With" in any text editor (e.g. TextEdit)
1. Add a line at the end with the following: `export PYTHONPATH=/Users/<username>/<path>` where `<username>` is your home folder name and `<path>` is the path to the folder where you saved the files
1. Save the file and press `shift` + `command` + `.` to hide hidden files again
1. Open a new terminal window for the changes to take effect
1. Type `echo $PYTHONPATH$` - you should see the path to your folder


### Windows:
1. Download the files from Github to a folder that will store your python code
1. From the taskbar search for "Advanced System Properties"
1. Click on "Environment variables" button at the bottom
1. In the top half of the window, under "User Variables" click on "New"
1. Enter `PYTHONPATH` as the variable name
1. Enter the path to your folder that you created in the value box
1. Click OK to save the new variable. Make sure it shows up in the list
1. From the search bar open up a command prompt by searching for `cmd`
1. At the prompt type `echo %PYTHONPATH%` - you should see the path to your folder

# Example Usage
A Jupyter notebook and example files are included in the repository. 

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
