import sys
from io import StringIO

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas import read_csv
from scipy import stats

st.set_page_config(
    page_title="Contours",
    page_icon=":large_blue_circle:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/jupedsim/jpscore",
        "Report a bug": "https://github.com/jupedsim/jpscore/issues",
        "About": "Extract geometries from trajectory files",
    },
)
st.sidebar.image("figs/jupedsim.png", use_column_width=True)
gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
repo = "https://github.com/chraibi/jupedsim-dashboard"
repo_name = f"[![Repo]({gh})]({repo})"
st.sidebar.markdown(repo_name, unsafe_allow_html=True)

def read_trajectory(input_file):
    data = read_csv(input_file, sep=r"\s+", dtype=np.float64, comment="#").values
    return data


trajectory_file = st.sidebar.file_uploader(
    "🚶 🚶‍♀️ Trajectory file ",
    type=["txt"],
    help="Load trajectory file",
)
if trajectory_file:
    data = read_trajectory(trajectory_file)

    dx = st.sidebar.slider("Grid size", 0.01, 0.5, 0.2, help="Grid size")
    ms = st.sidebar.slider("Marker size", 1, 100, 30, step=5, help="Marker size")
    e = st.sidebar.slider(
        "Buffer", 0.0, 5.0, 1.0, step=0.5, help="Add buffer around the trajectories"
    )
    pl = st.sidebar.empty()
    geominX = np.min(data[:, 2]) - e
    geominY = np.min(data[:, 3]) - e
    geomaxX = np.max(data[:, 2]) + e
    geomaxY = np.max(data[:, 3]) + e
    cm = 1 / 2.54
    w = geomaxX - geominX
    h = geomaxY - geominY
    fig, ax = plt.subplots(figsize=(w, h))
    xbins = np.arange(geominX - e, geomaxX + e, dx)
    ybins = np.arange(geominY - e, geomaxY + e, dx)
    X = data[:, 2]
    Y = data[:, 3]
    ret = stats.binned_statistic_2d(X, Y, None, "count", bins=[xbins, ybins])
    X = ret.x_edge + dx / 2
    Y = ret.y_edge + dx / 2
    x, y = np.meshgrid(X[:-1], Y[:-1])
    xx = x[ret.statistic.T != 0]
    yy = y[ret.statistic.T != 0]

    ax.set_xlim((geominX - e, geomaxX + e))
    ax.set_ylim((geominY - e, geomaxY + e))
    s = (ms * 72.0 / fig.dpi) ** 2
    ax.scatter(xx, yy, s=s, linewidth=0, color="white")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor("black")
    figname = "trajectories.png"
    fig.savefig(figname, dpi=100)

    # extract contours
    img = cv2.imread(figname, cv2.IMREAD_GRAYSCALE)
    fig1, ax1 = plt.subplots(1, 1, figsize=(w, h))
    # ax1.imshow(img, cmap="gray")
    _, binary_img = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Contour not detects")
        st.stop()
    elif len(contours) == 1:
        cnt = contours[0]
    else:
        cnt = contours[1]

    print(f"Number of Contours: {len(contours)}")
    final_image = cv2.drawContours(img, cnt, 0, (178, 34, 34), 2)
    x, y, w, h = cv2.boundingRect(cnt)
    cnt = np.array(cnt)
    sample = st.sidebar.slider(
        "Sample contour",
        1,
        int(len(cnt) / 10),
        5,
        help="Reduce number of points in contour",
    )

    # left button point
    st.sidebar.write("------")
    s = 0.2
    Options = []
    for i in range(5, -5, -1):
        Options.append(float(geominX - i * s))

    for i in range(-5, 6):
        Options.append(float(geomaxX + i * s))

    a, c = st.sidebar.select_slider(
        "⬅️  Shift horizontally ➡️",
        options=Options,
        format_func=lambda x: f"{x:.2f}",
        value=(geominX, geomaxX),
    )

    Options = []
    for i in range(5, -5, -1):
        Options.append(float(geominY - i * s))

    for i in range(-5, 6):
        Options.append(float(geomaxY + i * s))

    b, d = st.sidebar.select_slider(
        "⬆️    Shift vertically   ⬇️",
        options=Options,
        format_func=lambda x: f"{x:.2f}",
        value=(geominY, geomaxY),
    )

    st.sidebar.write("------")

    # remap coordinates
    Xc = a + (c - a) / w * (cnt[:, 0][:, 0] - x)
    Yc = b + (d - b) / h * (cnt[:, 0][:, 1] - y)

    ax1.plot(xx, yy, "s", ms=ms, color="white")
    ax1.plot(xx, yy, "s", ms=0.8, color="blue")
    ax1.plot(data[:, 2], data[:, 3], ".", color="gray", ms=0.5)
    ax1.plot(Xc[::sample], Yc[::sample], "red", lw=4)
    ax1.set_facecolor("black")
    ax1.set_xlim((geominX - e, geomaxX + e))
    ax1.set_ylim((geominY - e, geomaxY + e))
    st.pyplot(fig1)
    df = pd.DataFrame(
        {
            "X": Xc,
            "Y": Yc,
        }
    )

    # log messages
    st.write("------")
    c1, c2 = st.columns((1, 1))
    with c1:
        st.write(
            f"- Number of contours: {len(contours)}\n - Contour points: {len(Xc[::sample])}/{len(Xc)}"
        )

        st.dataframe(df.style.format(subset=["X", "Y"], formatter="{:.2f}"))

    with c2:
        # st.write(f"left buttom: ({a:.2f}, {b:.2f}), right up: ({c:.2f}, {d:.2f})")
        st.write(
            f"- Grid X: [{X[0]:.2f} - {X[-1]:.2f}], Grid Y: [{Y[0]:.2f} - {Y[-1]:.2f}]"
        )
        st.write(
            f"- Geo  X: [{geominX:.2f}, {geomaxX:.2f}], Geo  Y: [{geominY:.2f}, {geomaxY:.2f}]"
        )
        st.write(
            f"- Cont X: [{np.min(Xc):.2f}, {np.max(Xc):.2f}], Cont Y: [{np.min(Yc):.2f}, {np.max(Yc):.2f}]"
        )
        st.info(
            f"**Err  X: [{abs(geominX-np.min(Xc)):.2f}, {abs(geomaxX-np.max(Xc)):.2f}], Err Y: [{abs(geominY-np.min(Yc)):.2f}, {abs(geomaxY-np.max(Yc)):.2f}]**"
        )
