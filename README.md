
![Logo](https://i.ibb.co/xGdtf2F/Group-56.png)


# **SOT** - Map Analysis

**Brief:** Using a variety of methods to analyse a screen grab of an in game island map to determine which island the map refers to and to then send this data along with meta data concerning the island properties.

**Analysis Methods Used: **
A brief description of the types of methods used


# Application Installation

## Run Locally - Setup Virtual Environment

Clone the project

```bash
  git clone https://github.com/timhow38/sot-mapping
```

Open the project dir using "VSCode Community"

```powershell
  # Install Python >= 3.9
  py -m venv ./../sot-mapping
```

Setup Virtual Environment

```powershell
  # Powershell: 
  ./Scripts/activate

         or

  # Command Prompt: 
  Scripts/activate.bat
```

Install Dependencies. - **Only needed on first setup**

```powershell
  # Install dependencies for a closed env 
  py -m pip install -r requirements.txt
```


## Local Testing - Test / Generate
While Environment active type the following:

```py
  py tools/test_masking.py
```