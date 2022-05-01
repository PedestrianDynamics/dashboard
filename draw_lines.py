import io
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from Utilities import get_unit, read_trajectory

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_dimensions(data):
    e = 1
    geominX = np.min(data[:, 2]) - e
    geominY = np.min(data[:, 3]) - e
    geomaxX = np.max(data[:, 2]) + e
    geomaxY = np.max(data[:, 3]) + e
    return geominX, geomaxX, geominY, geomaxY


def get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY):
    scale = np.amin((geomaxX - geominX, geomaxY - geominY))
    scale_max = 50
    scale = min(scale_max, scale)
    scale = (1 - scale / scale_max) * 0.9 + scale / scale_max * 0.1
    scale = 0.3
    st.write(f"scale= {scale:.2f}")
    w = (geomaxX - geominX) * scale
    h = (geomaxY - geominY) * scale
    return w, h, scale


def plot_traj(ax, data, scale=1, shift_x=0, shift_y=0):
    pid = np.unique(data[:, 0])
    for ped in pid:
        pedd = data[data[:, 0] == ped]
        ax.plot(
            (pedd[::, 2] - shift_x) * scale,
            (pedd[::, 3] - shift_y) * scale,
            "-",
            color="black",
            lw=0.8,
        )


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def rotate(x, y, angle):
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


def process_lines(lines, h_dpi):
    """transform line's points to world coordinates

    :param line: the object for line
    :param h_dpi: height of image in dpi
    :returns: 2 points a 2 coordinates -> 4 values

    """
    left = np.array(lines["left"])
    top = np.array(lines["top"])
    # height = np.array(lines["height"])
    # width = np.array(lines["width"])
    scale_x = np.array(lines["scaleX"])
    scale_y = np.array(lines["scaleY"])
    angle = np.array(lines["angle"]) * np.pi / 180
    x1 = np.array(lines["x1"])
    x2 = np.array(lines["x2"])
    y1 = np.array(lines["y1"])
    y2 = np.array(lines["y2"])
    x1, y1 = rotate(x1, y1, angle)
    x2, y2 = rotate(x2, y2, angle)
    first_x = left + x1 * scale_x
    first_y = h_dpi - (top + y1 * scale_y)
    second_x = left + x2 * scale_x
    second_y = h_dpi - (top + y2 * scale_y)

    return first_x, first_y, second_x, second_y


def process_rects(rects, h_dpi):
    """transform rect's points to world coordinates

    :param rects: the object for rectangle
    :param h_dpi: height of image in dpi
    :returns: 4 points a 2 coordinates -> 8 values

    """
    left = np.array(rects["left"])
    top = np.array(rects["top"])
    scale_x = np.array(rects["scaleX"])
    scale_y = np.array(rects["scaleY"])
    height = np.array(rects["height"]) * scale_y
    width = np.array(rects["width"]) * scale_x
    angle = -np.array(rects["angle"]) * np.pi / 180
    # center_x = left + width / 2
    # center_y = top - height / 2
    # first
    first_x = left
    first_y = h_dpi - top

    # second
    x1, y1 = rotate(width, 0, angle)
    second_x = first_x + x1  # width
    second_y = first_y + y1
    # third
    x1, y1 = rotate(0, -height, angle)
    third_x = first_x + x1
    third_y = first_y + y1  # - height
    # forth
    x1, y1 = rotate(width, -height, angle)
    firth_x = first_x + x1  # width
    firth_y = first_y + y1
    # rotate

    return (
        first_x,
        first_y,
        second_x,
        second_y,
        third_x,
        third_y,
        firth_x,
        firth_y,
    )


def plot_lines(_inv, ax2, Xpoints, Ypoints, scale=1, shift_x=0, shift_y=0):
    """Plot lines

    Xpoints and Ypoints are sorted as follows:
    x_first_points followed by x_second_points
    y_first_points followed by y_second_points
    """
    aX = _inv.transform(Xpoints / scale)
    aY = _inv.transform(Ypoints / scale)
    num_points = aX.shape[0]
    num_points_half = int(num_points / 2)
    # plot resutls in real world coordinates
    for i in range(0, num_points_half):
        # st.write(aX[i:i+2])
        # st.write(aY[i:i+2])
        x = np.array([aX[i], aX[i + num_points_half]]) + shift_x
        y = np.array([aY[i], aY[i + num_points_half]]) + shift_y
        ax2.plot(x, y)


def write_geometry(first_x, first_y, second_x, second_y, _unit, geo_file):
    # ----------
    delta = 100 if _unit == "cm" else 1
    # --------
    # create_geo_header
    data = ET.Element("geometry")
    data.set("version", "0.8")
    data.set("caption", "experiment")
    data.set("unit", "m")  # jpsvis does not support another unit!
    # make room/subroom
    rooms = ET.SubElement(data, "rooms")
    room = ET.SubElement(rooms, "room")
    room.set("id", "0")
    room.set("caption", "room")
    subroom = ET.SubElement(room, "subroom")
    subroom.set("id", "0")
    subroom.set("caption", "subroom")
    subroom.set("class", "subroom")
    subroom.set("A_x", "0")
    subroom.set("B_y", "0")
    subroom.set("C_z", "0")
    # make lines horizontal/vertical
    # for x1, y1, x2, y2 in zip(first_x, first_y, second_x, second_y):
    #     st.info(f"before: {x1:.2f}, {y1:.2f}, {x2:.2f},{y2:.2f}")
    #     eps = 0.5
    #     if abs(x1-x2) < eps:
    #         x2 = x1

    #     if abs(y1-y2) < eps:
    #         y2 = y1

    #     st.info(f"after {x1:.2f}, {y1:.2f}, {x2:.2f},{y2:.2f}")

    # snap points
    p1_x = np.hstack((first_x[0], second_x[:]))
    p1_y = np.hstack((first_y[0], second_y[:]))
    p2_x = np.hstack((second_x, first_x[0]))
    p2_y = np.hstack((second_y, first_y[0]))
    for x1, y1, x2, y2 in zip(p1_x, p1_y, p2_x, p2_y):

        polygon = ET.SubElement(subroom, "polygon")
        polygon.set("caption", "wall")
        polygon.set("type", "internal")
        vertex = ET.SubElement(polygon, "vertex")
        vertex.set("px", f"{x1:.2f}")
        vertex.set("py", f"{y1:.2f}")
        vertex = ET.SubElement(polygon, "vertex")
        vertex.set("px", f"{x2:.2f}")
        vertex.set("py", f"{y2:.2f}")

    b_xml = ET.tostring(data, encoding="utf8", method="xml")
    with open(geo_file, "wb") as f:
        f.write(b_xml)


def main(trajectory_file):
    geo_file = ""
    data = read_trajectory(trajectory_file)
    geominX, geomaxX, geominY, geomaxY = get_dimensions(data)
    w, h, scale = get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY)
    st.write(f"w={w:.2f}, h={h:.2f}")
    st.write(f"geox [{geominX:.2f}, {geomaxX}] geoy [{geominY:.2f}, {geomaxY}]")
    # setup background figure
    width = w
    height = h
    fig, ax = plt.subplots(figsize=(width, height))
    fig.set_dpi(100)
    ax.set_xlim((0, width))
    ax.set_ylim((0, height))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    inv = ax.transData.inverted()
    img_width, img_height = bbox.width * fig.dpi, bbox.height * fig.dpi
    # st.info(f"width: {img_width}, height: {img_height}")
    plot_traj(ax, data, scale, geominX, geominY)

    c1, c2 = st.columns((1, 1))
    bg_img = fig2img(fig)
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("line", "rect", "transform"))
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#E80606")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=bg_img,
        update_streamlit=True,
        width=img_width,
        height=img_height,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")

        st.write(objects)
        if not objects.empty:
            lines = objects[objects["type"].values == "line"]
            rects = objects[objects["type"].values == "rect"]
            st.info(f"lines: {len(lines)}, rects: {len(rects)}")
            # result figure in world coordinates
            fig2, ax2 = plt.subplots()
            ax2.set_xlim((geominX, geomaxX))
            ax2.set_ylim((geominY, geomaxY))
            ax2.grid(alpha=0.3)
            # plot reference points
            plot_traj(ax2, data)

            if not lines.empty:
                first_x, first_y, second_x, second_y = process_lines(
                    lines, height * fig.get_dpi()
                )
                line_points_x = np.hstack((first_x, second_x))
                line_points_y = np.hstack((first_y, second_y))
                plot_lines(
                    inv, ax2, line_points_x, line_points_y, scale, geominX, geominY
                )

            if not rects.empty:
                (
                    first_x,
                    first_y,
                    second_x,
                    second_y,
                    third_x,
                    third_y,
                    firth_x,
                    firth_y,
                ) = process_rects(rects, height * fig.get_dpi())
                rect_points_x = np.hstack(
                    (
                        first_x,
                        second_x,
                        firth_x,
                        third_x,
                        second_x,
                        firth_x,
                        third_x,
                        first_x,
                    )
                )
                rect_points_y = np.hstack(
                    (
                        first_y,
                        second_y,
                        firth_y,
                        third_y,
                        second_y,
                        firth_y,
                        third_y,
                        first_y,
                    )
                )

                plot_lines(
                    inv, ax2, rect_points_x, rect_points_y, scale, geominX, geominY
                )

            st.pyplot(fig2)
            geo_file = "geo_" + trajectory_file.name.split(".")[0] + ".xml"
            write_geometry(
                first_x / scale / fig.dpi + geominX,
                first_y / scale / fig.dpi + geominY,
                second_x / scale / fig.dpi + geominX,
                second_y / scale / fig.dpi + geominY,
                "m",
                geo_file,
            )

    return geo_file


if __name__ == "__main__":
    trajectory_file = st.sidebar.file_uploader(
        "🚶 🚶‍♀️ Trajectory file ",
        type=["txt"],
        help="Load trajectory file",
    )
    if trajectory_file:
        file_xml = main(trajectory_file)
        if file_xml:
            with open(file_xml, encoding="utf-8") as f:
                st.sidebar.download_button(
                    "Download geometry", f, file_name=file_xml
                )
