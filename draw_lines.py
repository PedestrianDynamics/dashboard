import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
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


def plot_lines(_inv, ax2, Xpoints, Ypoints):
    """Plot lines

    Xpoints and Ypoints are sorted as follows:
    x_first_points followed by x_second_points
    y_first_points followed by y_second_points
    """
    aX = _inv.transform(Xpoints)
    aY = _inv.transform(Ypoints)
    num_points = aX.shape[0]
    num_points_half = int(num_points / 2)
    # plot resutls in real world coordinates
    for i in range(0, num_points_half):
        # st.write(aX[i:i+2])
        # st.write(aY[i:i+2])
        ax2.plot([aX[i], aX[i + num_points_half]], [aY[i], aY[i + num_points_half]])


def main():
    # setup background figure
    width = 2
    height = 3
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim((0, width))
    # ax.set_axis_off()
    ax.set_ylim((0, height))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    inv = ax.transData.inverted()
    img_width, img_height = bbox.width * fig.dpi, bbox.height * fig.dpi
    # st.info(f"width: {img_width}, height: {img_height}")

    # plot reference points
    ax.plot([1], [1], "o", ms=5)
    ax.plot([0.5], [1], "o", ms=5)
    ax.plot([0.5], [2], "o", ms=5)
    ax.plot([1], [2], "o", ms=5)

    c1, c2 = st.columns((1, 1))
    bg_img = fig2img(fig)
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("line", "rect", "transform"))
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    with c1:
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
            ax2.set_xlim((0, width))
            ax2.set_ylim((0, height))
            # set aspect ratio to 1
            ratio = 1.0
            x_left, x_right = ax2.get_xlim()
            y_low, y_high = ax2.get_ylim()
            ax2.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            ax2.grid(alpha=0.3)
            # plot reference points
            ax2.plot([1], [1], "o", ms=5)
            ax2.plot([0.5], [1], "o", ms=5)
            ax2.plot([0.5], [2], "o", ms=5)
            ax2.plot([1], [2], "o", ms=5)

            if not lines.empty:
                first_x, first_y, second_x, second_y = process_lines(
                    lines, height * fig.get_dpi()
                )
                line_points_x = np.hstack((first_x, second_x))
                line_points_y = np.hstack((first_y, second_y))
                plot_lines(inv, ax2, line_points_x, line_points_y)

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

                plot_lines(inv, ax2, rect_points_x, rect_points_y)

            with c2:
                st.pyplot(fig2)


if __name__ == "__main__":
    main()
