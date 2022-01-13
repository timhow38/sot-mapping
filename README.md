
![Logo](https://i.ibb.co/xGdtf2F/Group-56.png)


# **SOT** - Map Analysis

**Brief:** Using a variety of methods to analyse a screen grab of an in game island map to determine which island the map refers to and to then send this data along with meta data concerning the island properties.

**Analysis Methods Used: **
Identifies and isolates relevant features, creates binary masks and SIFT matching points, and creates a weighted match score against a cache of reference images


# Application Installation

## Run Locally - Setup Virtual Environment

Clone the project

```bash
  git clone https://github.com/timhow38/sot-mapping
```

Set up a venv in the same directory

```powershell
  # Install Python >= 3.9
  py -m venv ./../sot-mapping
```

Activate the new virtual environment

```powershell
  # Powershell: 
  ./Scripts/activate

         or

  # Command Prompt: 
  Scripts/activate.bat
```

Install Dependencies

```powershell
  # Install dependencies for a closed env 
  py -m pip install -r requirements.txt
```


## Run the program
With your environment active:

```py
  py tools/screen_thieves.py
```