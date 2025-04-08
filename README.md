# **OpenPods for Linux**
## Check your AirPods battery level on Linux

### What is it?
This is a Python 3.12 script, forked from [delphiki/AirStatus](https://github.com/delphiki/AirStatus) that allows you to check AirPods battery level from your terminal, as JSON output.

The code fully rafactored.

### Usage

```Bash
usage: main.py [-h] [--format {json,text}] [-o FILE]
```

Output will be stored in `output_file` if specified.

#### Output example 

```JSON
{
    "status": 1,
    "charge": {
        "left": 90,
        "right": -1,
        "case": -1
    },
    "charging_left": false,
    "charging_right": false,
    "charging_case": false,
    "model": "AirPods2",
    "date": "2025-04-08 17:59:41",
    "raw": "0719010f2022f98f01000409e411af2951df97aaf6f6bf5e0f9380"
}

```

### Installing as a service

Create the file `/etc/systemd/system/openpods_linux.service`

Enable and start the service:
```Bash
sudo systemctl enable openpods_linux --now
```

### Developer contact:
https://t.me/rus_99_pk

### Donate:
```
USDT (TRC20): TLX57npAx31fXsxvpe2ZAMfd71de6EJSbE
```
