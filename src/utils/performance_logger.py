import time
import functools
import os
import csv
from datetime import datetime

LOG_PATH = "outputs/performance_log.csv"


def log_runtime(task_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            elapsed = round(end - start, 4)

            os.makedirs("outputs", exist_ok=True)

            file_exists = os.path.exists(LOG_PATH)

            with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(["Timestamp", "Task", "Runtime_Seconds"])

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    task_name,
                    elapsed
                ])

            print(f"{task_name} completed in {elapsed} seconds")

            return result
        return wrapper
    return decorator