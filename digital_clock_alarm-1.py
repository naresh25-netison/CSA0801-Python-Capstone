import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import time

try:
    import winsound
except ImportError:
    winsound = None


class DigitalClockAlarm:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Clock and Alarm")
        self.root.geometry("520x360")
        self.root.resizable(False, False)

        self.alarm_time = None
        self.alarm_triggered = False

        title = tk.Label(
            root,
            text="DIGITAL CLOCK",
            font=("Arial", 22, "bold")
        )
        title.pack(pady=15)

        self.clock_label = tk.Label(
            root,
            text="00:00:00",
            font=("Arial", 48, "bold")
        )
        self.clock_label.pack(pady=5)

        self.date_label = tk.Label(
            root,
            text="",
            font=("Arial", 14)
        )
        self.date_label.pack(pady=5)

        alarm_frame = tk.Frame(root)
        alarm_frame.pack(pady=15)

        tk.Label(
            alarm_frame,
            text="Set Alarm (HH:MM:SS):",
            font=("Arial", 13)
        ).grid(row=0, column=0, padx=5)

        self.alarm_entry = tk.Entry(
            alarm_frame,
            width=12,
            font=("Arial", 13)
        )
        self.alarm_entry.grid(row=0, column=1, padx=5)

        tk.Button(
            root,
            text="Set Alarm",
            command=self.set_alarm,
            font=("Arial", 12, "bold")
        ).pack(pady=5)

        tk.Button(
            root,
            text="Cancel Alarm",
            command=self.cancel_alarm,
            font=("Arial", 11)
        ).pack(pady=3)

        self.status_label = tk.Label(
            root,
            text="No alarm set",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        self.update_clock()

    def update_clock(self):
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%A, %d %B %Y")

        self.clock_label.config(text=current_time)
        self.date_label.config(text=current_date)

        if self.alarm_time == current_time and not self.alarm_triggered:
            self.alarm_triggered = True
            self.ring_alarm()

        self.root.after(200, self.update_clock)

    def set_alarm(self):
        alarm = self.alarm_entry.get().strip()

        try:
            datetime.strptime(alarm, "%H:%M:%S")

            self.alarm_time = alarm
            self.alarm_triggered = False

            self.status_label.config(
                text="Alarm set for " + alarm
            )

            messagebox.showinfo(
                "Alarm",
                "Alarm set successfully for " + alarm
            )

        except ValueError:
            messagebox.showerror(
                "Invalid Time",
                "Enter time in HH:MM:SS format.\nExample: 18:30:00"
            )

    def cancel_alarm(self):
        self.alarm_time = None
        self.alarm_triggered = False
        self.status_label.config(text="No alarm set")

    def ring_alarm(self):
        self.status_label.config(text="ALARM RINGING!")

        if winsound:
            for i in range(5):
                winsound.Beep(1000, 500)
        else:
            self.root.bell()

        messagebox.showwarning(
            "ALARM",
            "Alarm time reached!"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitalClockAlarm(root)
    root.mainloop()
