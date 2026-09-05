# Digital Clock and Alarm using Python

A simple GUI-based digital clock and alarm application developed using Python.

## Features
- Displays current time in HH:MM:SS format
- Displays current date
- Allows the user to set an alarm
- Validates alarm time
- Allows alarm cancellation
- Rings an alarm when the specified time is reached

## Python Modules Used
- `tkinter` - GUI
- `datetime` - current date and time
- `winsound` - alarm sound on Windows

## How to Run

```bash
python digital_clock_alarm.py
```

Enter the alarm time in `HH:MM:SS` format, for example:

```text
18:30:00
```

Note: `winsound` is available on Windows. On other systems, the program uses the Tkinter alert bell.
