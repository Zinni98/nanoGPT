import time
import numpy as np
from pynvml import nvmlInit,\
    nvmlDeviceGetHandleByIndex,\
    nvmlDeviceGetTotalEnergyConsumption


def check_usage_interval(handle, base, step, precision=0.0000001):
    assert step < base
    if step < precision:
        return base
    for _ in range(int(base/step)):
        print(f"Trying: {base-step}")
        start = nvmlDeviceGetTotalEnergyConsumption(handle)
        time.sleep(base-step)
        stop = nvmlDeviceGetTotalEnergyConsumption(handle)
        diff = stop - start
        print(f"Diffrence: {diff}")
        base = base - step
        if diff == 0:
            return check_usage_interval(handle, base + step, step=step*(10**(-1)))
        elif base-step == 0:
            return check_usage_interval(handle, base, step=step*(10**(-1)))
    return 0


def main():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    avg = []
    for _ in range(100):
        avg.append(check_usage_interval(handle, 1.0, 0.1))
    avg = np.array(avg)
    print(avg.mean())


if __name__ == "__main__":
    main()

