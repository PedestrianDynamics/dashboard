import sys
from io import StringIO
import contextlib
import time
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas import read_csv
from scipy import stats
from PIL import Image
from io import StringIO

from Utilities import get_unit, read_trajectory
    
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


@contextlib.contextmanager
def profile(name):
    start_time = time.time()
    yield  # <-- your code will execute here
    total_time = time.time() - start_time
    st.info(f"{name}: {total_time * 1000.0:.4f} ms")




 
trajectory_file = st.sidebar.file_uploader(
    "🚶 🚶‍♀️ Trajectory file ",
    type=["txt"],
    help="Load trajectory file",
)
tpl = st.sidebar.empty()

if trajectory_file:
    start_time = time.time()
    stringio = StringIO(trajectory_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    unit = get_unit(string_data)
    print(f"unit {unit}")
    if unit == "cm":
        cm2m = 100
    elif unit == "m":
        cm2m = 1
    else:
        st.error(f"did not recognize unit {unit}")

        st.stop()

    data = read_trajectory(trajectory_file)/cm2m
    dx = st.sidebar.slider("Grid size", 0.1, 1.0, 0.5, help="Grid size")    
    e = st.sidebar.slider(
        "Buffer", 0.0, 5.0, 1.0, step=0.2, help="Add buffer around the trajectories"
    )
    eps = st.sidebar.slider("Precision", 0.001, 1., 0.1, step=0.1, help="Increase/descrease the number of points in contours")
    eps = eps/100
    pl = st.sidebar.empty()
    geominX = np.min(data[:, 2]) - e
    geominY = np.min(data[:, 3]) - e
    geomaxX = np.max(data[:, 2]) + e
    geomaxY = np.max(data[:, 3]) + e    
    w = (geomaxX - geominX)
    h = (geomaxY - geominY)
    print(f"w={w}, h={h}, {w*h}")
    fig, ax = plt.subplots(figsize=(w, h))
    xbins = np.arange(geominX - e, geomaxX + e, dx)
    ybins = np.arange(geominY - e, geomaxY + e, dx)
    X = data[:, 2]
    Y = data[:, 3]
    ret = stats.binned_statistic_2d(X, Y, None, "count", bins=[xbins, ybins])
    print("finished binned")
    X = ret.x_edge + dx / 2
    print("1")
    Y = ret.y_edge + dx / 2
    print("2")
    x, y = np.meshgrid(X[:-1], Y[:-1])
    print("3")
    arr = ret.statistic.T
    print("4")
    arr[arr > 0] = 255
    
    print("5")
    xx = x[ret.statistic.T != 0]
    yy = y[ret.statistic.T != 0]
    print("statistic")
    image = Image.fromarray(arr).convert('RGB')
    #image = image.resize((int(image.size[0]*20),int(image.size[1]*20))) #(int(w), int(h)))
    image = image.resize((int(w*100), int(h*100)))
    numpy_image = np.array(image)    
    width1, height1 = image.size
    print("w x h", width1, height1)
    #image.show()
    #st.image(image)
    # extract contours    
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    #img = cv2.flip(img, 0) # flip around x-axis
    
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(w, h))
    #xzax1.imshow(img, cmap="jet")
    
    _, binary_img = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #contours, hierarchy = cv2.findContours( close.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        st.warning("Contour not detects")
        st.stop()
    elif len(contours) == 1:
        cnt = contours[0]
    else:        
        cnt = sorted(contours, key=cv2.contourArea)[-2]

    # calc arclentgh
    arclen = cv2.arcLength(cnt, True)

    # do approx
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(w, h))
    perimeter = cv2.arcLength(cnt, True)
    approximatedShape = cv2.approxPolyDP(cnt.copy(), eps * perimeter, True)
    #final_image = cv2.drawContours(img, cnt, 0, (178, 34, 32), 2)
    #final_image = cv2.drawContours(img, approximatedShape, 0, (178, 34, 32), 2)
    
    
    #ax2.imshow(final_image)
    
    #st.pyplot(fig2)
    
    x, y, w, h = cv2.boundingRect(cnt)
    #total_time = time.time() - start_time
    #tpl.info(f"Time2: {total_time * 1000.0:.4f} ms")
    #start_time = time.time()
    cnt = np.array(cnt)
    # sample = st.sidebar.slider(
    #     "Sample contour",
    #     1,
    #     int(len(cnt) / 10),
    #     1,
    #     help="Reduce number of points in contour",
    # )
    sample=1
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

    #Cnt = cnt[:, 0]
    Cnt = approximatedShape[:, 0]
    Xc = a + (c - a) / w * (Cnt[:, 0] - x)
    Yc = b + (d - b) / h * (Cnt[:, 1] - y)

    Xc = np.hstack((Xc, Xc[0]))
    Yc = np.hstack((Yc, Yc[0]))
    #ax1.plot(xx, yy, "s", ms=ms, color="white")
    ax1.plot(xx, yy, "s", ms=3, color="red")
    pid = np.unique(data[:, 0])
    for ped in pid:
        d = data[data[:, 0] == ped]
        ax1.plot(d[::, 2], d[::, 3], "-", color="white", lw=0.8)
        
    ax1.plot(Xc[::sample], Yc[::sample], "o-", color="yellow", ms=10, lw=4)
    ax1.set_facecolor("black")
    ax1.set_xlim((geominX - e, geomaxX + e))
    ax1.set_ylim((geominY - e, geomaxY + e))
    c1, c2, c3 = st.columns((1, 0.5, 0.5))
    with c1:
        st.pyplot(fig1)
    df = pd.DataFrame(
        {
            "CX": Xc,
            "CY": Yc,
        }
    )

    # log messages
    st.write("------")
    
    with c3:
        st.write(
            f"- Number of contours: {len(contours)}\n - Contour points: {len(Xc)}"
        )

        st.dataframe(df.style.format(subset=["CX", "CY"], formatter="{:.2f}"))

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
        tpl2 = st.empty()
        st.info(
            f"**Err  X: [{abs(geominX-np.min(Xc)):.2f}, {abs(geomaxX-np.max(Xc)):.2f}], Err Y: [{abs(geominY-np.min(Yc)):.2f}, {abs(geomaxY-np.max(Yc)):.2f}]**"
        )

    total_time = time.time() - start_time
    tpl2.info(f"Time: {total_time * 1000.0:.4f} ms")
