from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------- CONFIG ---------
INPUT = "Random_test_python/Summary/output.csv"
OUTDIR = Path("plots")
OUTDIR.mkdir(exist_ok=True)


# --------- LOAD ----------
df = pd.read_csv(INPUT, header=None)

# Detect format
if df.shape[1] == 22:
    mode = "patch"
    df.columns = [
        "cv_all",
        "cv_high",
        "cv_low",
        "ang_all",
        "ang_high",
        "ang_low",
        "dir_all",
        "dir_high",
        "dir_low",
        "mag_all",
        "mag_high",
        "mag_low",
        "n_high",
        "lx",
        "ly",
        "shiftX",
        "shiftY",
        "lambdaX",
        "lambdaY",
        "low_conc",
        "rep1",
        "rep2",
    ]
else:
    mode = "random"
    df.columns = [
        "cv_all",
        "cv_high",
        "cv_low",
        "ang_all",
        "ang_high",
        "ang_low",
        "dir_all",
        "dir_high",
        "dir_low",
        "mag_all",
        "mag_high",
        "mag_low",
        "n_high",
        "low_conc",
    ]


# --------- UTIL ----------
def savefig(name):
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{name}.png", dpi=200)
    plt.close()


# --------- CORE PLOTS ----------

# CV vs low_conc
plt.scatter(df["low_conc"], df["cv_all"], s=10)
plt.xlabel("low")
plt.ylabel("cv")
plt.title("cv vs low")
savefig("cv_vs_low")


# magnitude vs low_conc
plt.scatter(df["low_conc"], df["mag_all"], s=10)
plt.xlabel("low")
plt.ylabel("mag")
plt.title("mag vs low")
savefig("mag_vs_low")


# angle deviation vs low_conc
plt.scatter(df["low_conc"], df["ang_all"], s=10)
plt.xlabel("low")
plt.ylabel("angle")
plt.title("angle vs low")
savefig("angle_vs_low")


# --------- HIGH vs LOW comparison ----------

plt.scatter(df["cv_high"], df["cv_low"], s=10)
plt.xlabel("cv high")
plt.ylabel("cv low")
plt.title("high vs low")
savefig("cv_high_vs_low")


plt.scatter(df["mag_high"], df["mag_low"], s=10)
plt.xlabel("mag high")
plt.ylabel("mag low")
plt.title("mag high vs low")
savefig("mag_high_vs_low")


# --------- DISTRIBUTION PLOTS ----------

plt.hist(df["cv_all"], bins=40)
plt.title("cv dist")
savefig("cv_dist")


plt.hist(df["mag_all"], bins=40)
plt.title("mag dist")
savefig("mag_dist")


# --------- CORRELATIONS ----------

corr = df.corr(numeric_only=True)

plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.title("corr")
plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
plt.yticks(range(len(corr)), corr.columns, fontsize=6)
savefig("correlation")


# --------- PATCH-SPECIFIC ----------
if mode == "patch":
    # cluster size vs CV
    size = df["lx"] * df["ly"]
    plt.scatter(size, df["cv_all"], s=10)
    plt.xlabel("size")
    plt.ylabel("cv")
    plt.title("size vs cv")
    savefig("size_vs_cv")

    # shift vs CV
    shift = df["shiftX"] + df["shiftY"]
    plt.scatter(shift, df["cv_all"], s=10)
    plt.xlabel("shift")
    plt.ylabel("cv")
    plt.title("shift vs cv")
    savefig("shift_vs_cv")

    # spacing vs CV
    spacing = df["lambdaX"] * df["lambdaY"]
    plt.scatter(spacing, df["cv_all"], s=10)
    plt.xlabel("spacing")
    plt.ylabel("cv")
    plt.title("spacing vs cv")
    savefig("spacing_vs_cv")

    # heatmap: lx vs ly
    pivot = df.pivot_table(index="lx", columns="ly", values="cv_all", aggfunc="mean")
    plt.imshow(pivot, origin="lower")
    plt.colorbar()
    plt.title("lx ly")
    savefig("lx_ly_heat")


# --------- RANDOM-SPECIFIC ----------
if mode == "random":
    # fraction high vs CV
    frac_high = df["n_high"] / df["n_high"].max()
    plt.scatter(frac_high, df["cv_all"], s=10)
    plt.xlabel("frac high")
    plt.ylabel("cv")
    plt.title("frac vs cv")
    savefig("frac_vs_cv")

    # phase-like plot
    plt.scatter(frac_high, df["mag_all"], s=10)
    plt.xlabel("frac high")
    plt.ylabel("mag")
    plt.title("frac vs mag")
    savefig("frac_vs_mag")


# --------- ADVANCED (extra) ----------

# CV vs magnitude
plt.scatter(df["mag_all"], df["cv_all"], s=10)
plt.xlabel("mag")
plt.ylabel("cv")
plt.title("mag vs cv")
savefig("mag_vs_cv")


# angle vs magnitude
plt.scatter(df["mag_all"], df["ang_all"], s=10)
plt.xlabel("mag")
plt.ylabel("angle")
plt.title("mag vs angle")
savefig("mag_vs_angle")


# PCA-like projection
X = df.select_dtypes(float).fillna(0).values
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
U, S, Vt = np.linalg.svd(X, full_matrices=False)

plt.scatter(U[:, 0], U[:, 1], s=10)
plt.title("pca")
savefig("pca")


print(f"Done. Plots saved in {OUTDIR}")
