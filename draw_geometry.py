import collections
import io
import timeit
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

import lovely_logger as logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from Utilities import get_time, get_unit, read_trajectory


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    reparsed = parseString(elem)
    return reparsed.toprettyxml(indent="\t")


def get_dimensions(data):
    e = 1
    geominX = np.min(data[:, 2]) - e
    geominY = np.min(data[:, 3]) - e
    geomaxX = np.max(data[:, 2]) + e
    geomaxY = np.max(data[:, 3]) + e
    return geominX, geomaxX, geominY, geomaxY


def get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY):
    scale = np.amin((geomaxX - geominX, geomaxY - geominY))
    scale_max = 20
    scale = min(scale_max, scale)
    scale = (1 - scale / scale_max) * 0.9 + scale / scale_max * 0.1
    # scale = 0.3
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


def plot_lines(_inv, ax2, Xpoints, Ypoints, color, scale=1, shift_x=0, shift_y=0):
    """Plot lines

    Xpoints and Ypoints are sorted as follows:
    x_first_points followed by x_second_points
    y_first_points followed by y_second_points

    This function scales and shifts the data.
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
        ax2.plot(x, y, color=color)


def write_geometry(
    first_x,
    first_y,
    second_x,
    second_y,
    mfirst_x,
    mfirst_y,
    msecond_x,
    msecond_y,
    rect_points_xml,
    _unit,
    geo_file,
):
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
    # snap points
    if len(first_x) > 1:
        p1_x = np.hstack((first_x[0], second_x)) * delta
        p1_y = np.hstack((first_y[0], second_y)) * delta
        p2_x = np.hstack((second_x, first_x[0])) * delta
        p2_y = np.hstack((second_y, first_y[0])) * delta
    else:
        p1_x = first_x * delta
        p1_y = first_y * delta
        p2_x = second_x * delta
        p2_y = second_y * delta

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

    # add measurement areas
    m_areas = ET.SubElement(data, "measurement_areas")
    m_areas.set("unit", _unit)
    for i in rect_points_xml.keys():
        area_B = ET.SubElement(m_areas, "area_B")
        rect_x = np.array(rect_points_xml[i]["x"]) * delta
        rect_y = np.array(rect_points_xml[i]["y"]) * delta
        area_B.set("id", str(i + 1))
        area_B.set("type", "BoundingBox")
        area_B.set("zPos", "None")
        for (x, y) in zip(rect_x, rect_y):
            vertex = ET.SubElement(area_B, "vertex")
            vertex.set("px", f"{x:.2f}")
            vertex.set("py", f"{y:.2f}")

    if mfirst_x is not None:
        i = 0
        for x1, y1, x2, y2 in zip(mfirst_x, mfirst_y, msecond_x, msecond_y):
            i += 1
            area_L = ET.SubElement(m_areas, "area_L")
            area_L.set("id", str(i))
            area_L.set("type", "Line")
            area_L.set("zPos", "None")
            start = ET.SubElement(area_L, "start")
            end = ET.SubElement(area_L, "end")
            start.set("px", f"{x1:.2f}")
            start.set("py", f"{y1:.2f}")
            end.set("px", f"{x2:.2f}")
            end.set("py", f"{y2:.2f}")

    b_xml = ET.tostring(data, encoding="utf8", method="xml")
    b_xml = prettify(b_xml)

    with open(geo_file, "w") as f:
        f.write(b_xml)

    return b_xml


def main(trajectory_file):
    geo_file = ""
    m_lines = []
    stringio = io.StringIO(trajectory_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    if string_data != st.session_state.old_data:
        st.session_state.old_data = string_data
        new_data = True
        logging.info("Loading new trajectory data")
    else:
        logging.info("Trajectory data existing")
        new_data = False

    unit = get_unit(string_data)
    logging.info(f"unit {unit}")
    if unit not in ["cm", "m"]:
        st.error(f"did not recognize unit from file: {unit}")
        unit = st.sidebar.radio(
            "What is the unit of the trajectories?",
            ["cm", "m"],
            help="Choose the unit of the original trajectories. Data in the app will be converted to meter",
        )
        st.write(
            "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
            unsafe_allow_html=True,
        )
    if unit == "cm":
        cm2m = 100
    elif unit == "m":
        cm2m = 1

    global debug
    debug = st.sidebar.checkbox("Show", help="plot result with ticks and show xml")
    st.sidebar.write("----")
    if new_data:
        data = read_trajectory(trajectory_file) / cm2m
        geominX, geomaxX, geominY, geomaxY = get_dimensions(data)
        width, height, scale = get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY)
        st.session_state.data = data
        # setup background figure
        fig, ax = plt.subplots(figsize=(width, height))
        fig.set_dpi(100)
        ax.set_xlim((0, width))
        ax.set_ylim((0, height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        inv = ax.transData.inverted()
        # st.info(f"width: {img_width}, height: {img_height}")
        plot_traj(ax, data, scale, geominX, geominY)
        major_ticks_top_x = np.linspace(0, width, 5)
        major_ticks_top_y = np.linspace(0, height, 5)
        minor_ticks_top_x = np.linspace(0, width, 40)
        minor_ticks_top_y = np.linspace(0, height, 40)
        major_ticks_bottom_x = np.linspace(0, width, 20)
        major_ticks_bottom_y = np.linspace(0, height, 20)
        ax.set_xticks(major_ticks_top_x)
        ax.set_yticks(major_ticks_top_y)
        ax.set_xticks(minor_ticks_top_x, minor=True)
        ax.set_yticks(minor_ticks_top_y, minor=True)
        ax.grid(which="major", alpha=0.6)
        ax.grid(which="minor", alpha=0.3)
        ax.set_xticks(major_ticks_bottom_x)
        ax.set_yticks(major_ticks_bottom_y)
        ax.set_title("Subplot 2")
        ax.grid()
        bg_img = fig2img(fig)
        st.session_state.bg_img = bg_img
        st.session_state.ax = ax
        st.session_state.fig = fig
    else:
        bg_img = st.session_state.bg_img
        data = st.session_state.data
        geominX, geomaxX, geominY, geomaxY = get_dimensions(st.session_state.data)
        width, height, scale = get_scaled_dimensions(geominX, geomaxX, geominY, geomaxY)

    fig = st.session_state.fig
    ax = st.session_state.ax
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    img_width, img_height = bbox.width * fig.dpi, bbox.height * fig.dpi
    inv = ax.transData.inverted()

    drawing_mode = st.sidebar.radio(
        "Drawing tool:", ("wall", "m_area", "m_line", "transform")
    )
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    # stroke_colocr = c1.color_picker("Stroke color hex: ", "#E80606")

    if drawing_mode == "m_line":
        stroke_color = st.session_state.stroke_mline

    if drawing_mode in ["wall", "m_area", "transform"]:
        stroke_color = st.session_state.stroke_wall

    if drawing_mode in ["wall", "m_line"]:
        drawing_mode = "line"

    if drawing_mode == "m_area":
        drawing_mode = "rect"

    canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#eee",
        background_image=bg_img,
        update_streamlit=True,
        width=img_width,
        height=img_height,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    global download_pl
    download_pl = st.empty()
    if debug:
        st.info(
            f"""
            w: {width:.2f}, h: {height:.2f} \n
            scale: {scale:.2f} \n
            x-axis: [{geominX:.2f}, {geomaxX:.2f}]\n
            y-axis: [{geominY:.2f}, {geomaxY:.2f}]"""
        )
    if canvas.json_data is not None:
        objects = pd.json_normalize(canvas.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")

        if not objects.empty:
            cond_mline = np.logical_and(
                objects["type"].values == "line",
                objects["stroke"].values == st.session_state.stroke_mline,
            )
            cond_wall = np.logical_and(
                objects["type"].values == "line",
                objects["stroke"].values == st.session_state.stroke_wall,
            )
            mlines = objects[cond_mline]
            lines = objects[cond_wall]
            rects = objects[objects["type"].values == "rect"]
            # result figure in world coordinates

            if not st.session_state.first_plot_traj:
                fig2, ax2 = plt.subplots()
                ax2.set_xlim((geominX, geomaxX))
                ax2.set_ylim((geominY, geomaxY))
                ax2.grid(alpha=0.3)
                plot_traj(ax2, data)
                st.session_state.fig2 = fig2
                st.session_state.ax2 = ax2
                st.session_state.first_plot_traj = True

            else:
                fig2 = st.session_state.fig2
                ax2 = st.session_state.ax2

            wfirst_x = np.array([])
            if not lines.empty:
                wfirst_x, wfirst_y, wsecond_x, wsecond_y = process_lines(
                    lines, height * fig.get_dpi()
                )
                wline_points_x = np.hstack((wfirst_x, wsecond_x))
                wline_points_y = np.hstack((wfirst_y, wsecond_y))
                if debug:
                    plot_lines(
                        inv,
                        ax2,
                        wline_points_x,
                        wline_points_y,
                        st.session_state.stroke_wall,
                        scale,
                        geominX,
                        geominY,
                    )
            # --------------------
            mfirst_x = np.array([])
            if not mlines.empty:
                mfirst_x, mfirst_y, msecond_x, msecond_y = process_lines(
                    mlines, height * fig.get_dpi()
                )
                mline_points_x = np.hstack((mfirst_x, msecond_x))
                mline_points_y = np.hstack((mfirst_y, msecond_y))
                if debug:
                    plot_lines(
                        inv,
                        ax2,
                        mline_points_x,
                        mline_points_y,
                        st.session_state.stroke_mline,
                        scale,
                        geominX,
                        geominY,
                    )
                    # --------------------
            rect_points_xml = collections.defaultdict(dict)
            if not rects.empty:
                (
                    rfirst_x,
                    rfirst_y,
                    rsecond_x,
                    rsecond_y,
                    rthird_x,
                    rthird_y,
                    rfirth_x,
                    rfirth_y,
                ) = process_rects(rects, height * fig.get_dpi())
                rect_points_x = np.hstack(
                    (
                        rfirst_x,
                        rsecond_x,
                        rfirth_x,
                        rthird_x,
                        rsecond_x,
                        rfirth_x,
                        rthird_x,
                        rfirst_x,
                    )
                )
                rect_points_y = np.hstack(
                    (
                        rfirst_y,
                        rsecond_y,
                        rfirth_y,
                        rthird_y,
                        rsecond_y,
                        rfirth_y,
                        rthird_y,
                        rfirst_y,
                    )
                )
                i = 0
                for x1, x2, x3, x4, y1, y2, y3, y4 in zip(
                    rfirst_x,
                    rsecond_x,
                    rthird_x,
                    rfirth_x,
                    rfirst_y,
                    rsecond_y,
                    rthird_y,
                    rfirth_y,
                ):

                    rect_points_xml[i]["x"] = [
                        x1 / scale / fig.dpi + geominX,
                        x2 / scale / fig.dpi + geominX,
                        x3 / scale / fig.dpi + geominX,
                        x4 / scale / fig.dpi + geominX,
                    ]
                    rect_points_xml[i]["y"] = [
                        y1 / scale / fig.dpi + geominY,
                        y2 / scale / fig.dpi + geominY,
                        y3 / scale / fig.dpi + geominY,
                        y4 / scale / fig.dpi + geominY,
                    ]
                    i += 1

                if debug:
                    plot_lines(
                        inv,
                        ax2,
                        rect_points_x,
                        rect_points_y,
                        st.session_state.stroke_mline,
                        scale,
                        geominX,
                        geominY,
                    )

            if debug:
                st.pyplot(fig2)

            geo_file = "geo_" + trajectory_file.name.split(".")[0] + ".xml"

            if wfirst_x.size != 0:
                if mfirst_x.size != 0:
                    b_xml = write_geometry(
                        wfirst_x / scale / fig.dpi + geominX,
                        wfirst_y / scale / fig.dpi + geominY,
                        wsecond_x / scale / fig.dpi + geominX,
                        wsecond_y / scale / fig.dpi + geominY,
                        mfirst_x / scale / fig.dpi + geominX,
                        mfirst_y / scale / fig.dpi + geominY,
                        msecond_x / scale / fig.dpi + geominX,
                        msecond_y / scale / fig.dpi + geominY,
                        rect_points_xml,
                        unit,
                        geo_file,
                    )
                else:
                    b_xml = write_geometry(
                        wfirst_x / scale / fig.dpi + geominX,
                        wfirst_y / scale / fig.dpi + geominY,
                        wsecond_x / scale / fig.dpi + geominX,
                        wsecond_y / scale / fig.dpi + geominY,
                        None,
                        None,
                        None,
                        None,
                        rect_points_xml,
                        unit,
                        geo_file,
                    )
                if debug:
                    st.code(b_xml, language="xml")
                    st.write(objects)

    return geo_file


def set_state_variables():
    if "old_data" not in st.session_state:
        st.session_state.old_data = ""

    if "data" not in st.session_state:
        st.session_state.data = np.array([])

    if "unit" not in st.session_state:
        st.session_state.unit = "m"

    if "bg_img" not in st.session_state:
        st.session_state.bg_img = None

    if "ax" not in st.session_state:
        st.session_state.ax = None

    if "fig" not in st.session_state:
        st.session_state.fig = None

    if "stroke_wall" not in st.session_state:
        st.session_state.stroke_wall = "#E80606"

    if "stroke_mline" not in st.session_state:
        st.session_state.stroke_mline = "#060EE8"

    if "first_plot_traj" not in st.session_state:
        st.session_state.first_plot_traj = False

    if "ax2" not in st.session_state:
        st.session_state.ax2 = None

    if "fig2" not in st.session_state:
        st.session_state.fig2 = None


if __name__ == "__main__":
    st.sidebar.image("figs/jupedsim.png", use_column_width=True)
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/chraibi/jupedsim-dashboard"
    repo_name = f"[![Repo]({gh})]({repo})"
    st.sidebar.markdown(repo_name, unsafe_allow_html=True)
    trajectory_file = st.sidebar.file_uploader(
        "🚶 🚶‍♀️ Trajectory file ",
        type=["txt"],
        help="Load trajectory file",
    )
    set_state_variables()
    if trajectory_file:
        time_start = timeit.default_timer()
        file_xml = main(trajectory_file)
        time_end = timeit.default_timer()
        msg_time = get_time(time_end - time_start)
        if debug:
            st.sidebar.info(f":clock8: Finished in {msg_time}")

        if file_xml:
            st.sidebar.write("-----")
            with open(file_xml, encoding="utf-8") as f:
                download_pl.download_button("Download geometry", f, file_name=file_xml)
