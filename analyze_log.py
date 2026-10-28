import sys
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scale = 0.060975611
rates = [75, 75, 70]
rc_rate = 1.08
axes = ["Roll", "Pitch", "Yaw"]
intervals = {0: [(12.0, 14.5), (25.0, 28.0)],
             1: [(5.8, 8.0), (10.0, 12.0)],
             2: [(16.5, 22.0), (60.0, 64.0)]}


def read_csv(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    t = (df["time (us)"].values - df["time (us)"].values[0]) / 1e6
    return df, t


def get_setpoint(rc, rate):
    f = rc / 500.0
    ang = 200 * rc_rate * f
    k = 1 / np.clip(1 - np.abs(f) * rate / 100.0, 0.01, 1)
    return ang * k


def metrics(t, sp_all, gyro_all, fs, t0, t1):
    b, a = signal.butter(2, 25 / (fs / 2))
    gl_all = signal.filtfilt(b, a, gyro_all)
    m = (t >= t0) & (t < t1)
    sp, gl, g = sp_all[m], gl_all[m], gyro_all[m]
    cc = []
    for l in range(0, int(0.15 * fs)):
        x = sp[:len(sp) - l] if l else sp
        cc.append(np.corrcoef(x, gl[l:])[0, 1])
    delay = int(np.nanargmax(cc)) / fs * 1000
    i = np.argmax(np.abs(sp))
    sg = np.sign(sp[i])
    over = ((gl * sg).max() - abs(sp[i])) / abs(sp[i]) * 100
    err = sp - gl
    return {"peak": sp[i], "delay_ms": delay, "overshoot": over,
            "mean_err": np.mean(np.abs(err)), "noise": np.std(g - gl)}


def heatmap(g, thr, fs, name):
    n = 512
    w = np.hanning(n)
    bins = np.arange(1260, 1561, 20)
    h = np.zeros((len(bins) - 1, n // 2 + 1))
    cnt = np.zeros(len(bins) - 1)
    for s in range(int(5.5 * fs), len(g) - n, 96):
        k = np.searchsorted(bins, thr[s + n // 2]) - 1
        if 0 <= k < len(cnt):
            seg = g[s:s + n] - g[s:s + n].mean()
            h[k] += np.abs(np.fft.rfft(seg * w)) ** 2 / (fs * (w ** 2).sum())
            cnt[k] += 1
    h = h / np.maximum(cnt, 1)[:, None]
    h[cnt < 6] = np.nan
    plt.figure(figsize=(7, 5))
    plt.imshow(10 * np.log10(h).T, origin="lower", aspect="auto",
               extent=[bins[0], bins[-1], 0, fs / 2], cmap="inferno", vmin=-12, vmax=16)
    for f in (90, 100, 160, 260):
        plt.axhline(f, ls="--", lw=0.8)
    plt.xlabel("throttle")
    plt.ylabel("Hz")
    plt.colorbar(label="dB")
    plt.savefig(name, dpi=110, bbox_inches="tight")


def main(path):
    df, t = read_csv(path)
    fs = 1 / np.median(np.diff(t))
    print(t[-1], fs)
    thr = df["rcCommand[3]"].values
    for ax in range(3):
        g = df["gyroADC[%d]" % ax].values * scale
        sp = get_setpoint(df["rcCommand[%d]" % ax].values, rates[ax])
        p = df["axisP[%d]" % ax].values
        c = np.linalg.lstsq(np.c_[sp, g], p, rcond=None)[0]
        print(axes[ax], c)
        for t0, t1 in intervals[ax]:
            r = metrics(t, sp, g, fs, t0, t1)
            print("  ", t0, t1, {k: round(float(v), 1) for k, v in r.items()})
    heatmap(df["gyroADC[0]"].values * scale, thr, fs, "heatmap_roll.png")


main(sys.argv[1])
