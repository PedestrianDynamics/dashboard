import lovely_logger as logging
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots


def plot_NT(Frames, Nums, fps):
    logging.info("plot NT-curve")
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=["N-T"], x_title="Time / s", y_title="Number of pedestrians"
    )
    for i, frames in Frames.items():
        nums = Nums[i]
        if not frames:
            continue

        # extend the lines to 0
        if frames[0] > 0:
            frames = np.hstack(([0], frames))
            nums = np.hstack(([0], nums))

        trace = go.Scatter(
            x=np.array(frames) / fps,
            y=nums,
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            marker=dict(size=1),
            line=dict(width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    # eps = 0.5
    # fig.update_xaxes(range=[xmin/fps - eps, xmax/fps + eps])
    # fig.update_yaxes(range=[ymin - eps, ymax + eps], autorange=False)
    # fig.update_layout(
    #     width=500,
    #     height=500,
    # )
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    #     autorange=True,
    # )
    st.plotly_chart(fig, use_container_width=True)


def plot_flow(Frames, Nums, fps):
    logging.info("plot flow-curve")
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=["Flow"], x_title="Time / s", y_title="J / 1/s"
    )
    for i, frames in Frames.items():
        nums = Nums[i]
        if not frames:
            continue

        times = np.array(frames) / fps
        J = nums / times
        trace = go.Scatter(
            x=np.hstack(([0, times[0]], times)),
            y=np.hstack(([0, 0], J)),
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            marker=dict(size=1),
            line=dict(width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    # fig.update_layout(
    #     width=500,
    #     height=500,
    # )
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    #     autorange=True,
    # )
    st.plotly_chart(fig, use_container_width=True)


def plot_peds_inside(frames, peds_inside, fps):
    logging.info("plot peds inside")
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=["Evacuation rate"], x_title="Time / s", y_title="Number of Pedestrians"
    )    
    times = frames / fps
    trace = go.Scatter(
        x=times,
        y=peds_inside,
        #mode="none",
        #fill='tozeroy',
        mode="lines",
        showlegend=False,
        
        #name=f"ID: {i}",
        marker=dict(size=1),
        line=dict(width=1),
    )
    fig.append_trace(trace, row=1, col=1)

    # fig.update_layout(
    #     width=500,
    #     height=500,
    # )
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    #     autorange=True,
    # )
    st.plotly_chart(fig, use_container_width=True)


def plot_agent_speed(pid, frames, speed_agent, fps):
    logging.info(f"plot agent speed {pid}")
    times = frames / fps
    trace = go.Scatter(
        x=times,
        y=speed_agent,
        mode="lines",
        showlegend=True,
        name=f"P: {pid:.0f}",
        marker=dict(size=1),
        line=dict(width=1),
    )
    return trace


def plot_trajectories(data, geo_walls, transitions, min_x, max_x, min_y, max_y):
    fig = make_subplots(rows=1, cols=1)
    peds = np.unique(data[:, 0])
    for ped in peds:
        d = data[data[:, 0] == ped]
        c = d[:, -1]
        trace = go.Scatter(
            x=d[:, 2],
            y=d[:, 3],
            mode="lines",
            showlegend=False,
            name=f"{ped:0.0f}",
            marker=dict(size=1, color=c),
            line=dict(color="gray", width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    for gw in geo_walls.keys():
        trace = go.Scatter(
            x=geo_walls[gw][:, 0],
            y=geo_walls[gw][:, 1],
            showlegend=False,
            mode="lines",
            line=dict(color="black", width=2),
        )
        fig.append_trace(trace, row=1, col=1)

    for i, t in transitions.items():
        xm = np.sum(t[:, 0]) / 2
        ym = np.sum(t[:, 1]) / 2
        length = np.sqrt(np.diff(t[:, 0]) ** 2 + np.diff(t[:, 1]) ** 2)
        offset = 0.1 * length[0]
        logging.info(f"offset transition {offset}")
        trace = go.Scatter(
            x=t[:, 0],
            y=t[:, 1],
            showlegend=False,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(color="black", size=5),
        )
        trace_text = go.Scatter(
            x=[xm + offset],
            y=[ym + offset],
            text=f"{i}",
            textposition="middle center",
            showlegend=False,
            mode="markers+text",
            marker=dict(color="red", size=0.1),
            textfont=dict(color="red", size=18),
        )
        fig.append_trace(trace, row=1, col=1)
        fig.append_trace(trace_text, row=1, col=1)

    eps = 1
    fig.update_layout(
        width=500,
        height=500,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        range=[min_y - eps, max_y + eps],
        # autorange=True,
    )
    fig.update_xaxes(
        #      #scaleanchor="y",
        #     # scaleratio=1,
        range=[min_x - eps, max_x + eps],
        #     autorange=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0], _geometry_wall[gw][:, 1], color="white", lw=2)


def plot_profile_and_geometry(
    geominX,
    geomaxX,
    geominY,
    geomaxY,
    geometry_wall,
    data,
    interpolation,
    cmap,
    label,
    title,
    vmin=None,
    vmax=None,
):
    """Plot profile + geometry for 3D data


    if vmin or vmax is None, extract values from <data>
    """

    if vmin is None or vmax is None:
        vmin = np.min(data)
        vmax = np.max(data)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        data,
        cmap=cmap,
        interpolation=interpolation,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[geominX, geomaxX, geominY, geomaxY],
    )
    plot_geometry(ax, geometry_wall)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, rotation=90, labelpad=15, fontsize=15)
    st.pyplot(fig)
