import lovely_logger as logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots
from shapely.geometry import LineString, Point
from scipy import stats

from Utilities import survival


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def show_trajectories_table(data):
    headerColor = "grey"
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>ID</b>", "<b>Frame</b>", "<b>X</b>", "<b>Y</b>"],
                    fill_color=headerColor,
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[data[:, 0], data[:, 1], data[:, 2], data[:, 3]],
                ),
            )
        ]
    )
    return fig


@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={go.Figure: lambda _: None})
def plot_NT(Frames, Nums, Nums_positiv, Nums_negativ, fps):
    logging.info("plot NT-curve")
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["<b>N-T</b>"],
        x_title="Time / s",
        y_title="Number at line",
    )
    maxx = -1
    traces = []
    for i, _frames in Frames.items():
        nums = Nums[i]
        nums_positiv = Nums_positiv[i]
        nums_negativ = Nums_negativ[i]
        frames = _frames[:, 1]
        if not frames.size:
            continue

        # extend the lines to 0
        if frames[0] > 0:
            frames = np.hstack(([0], frames))
            nums_positiv = np.hstack(([0], nums_positiv))
            nums_negativ = np.hstack(([0], nums_negativ))
            nums = np.hstack(([0], nums))
            if maxx < np.max(frames):
                maxx = np.max(frames)

        trace = go.Scatter(
            x=np.array(frames) / fps,
            y=nums,
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            line=dict(width=3),
        )
        if nums_positiv.any() and nums_negativ.any():
            trace_positiv = go.Scatter(
                x=np.array(frames) / fps,
                y=nums_positiv,
                mode="lines",
                showlegend=True,
                name=f"ID: {i}+",
                line=dict(width=3),
            )
            traces.append(trace_positiv)
            trace_negativ = go.Scatter(
                x=np.array(frames) / fps,
                y=nums_negativ,
                mode="lines",
                showlegend=True,
                name=f"ID: {i}-",
                line=dict(width=3),
            )
            traces.append(trace_negativ)

        traces.append(trace)


    for trace in traces:
        fig.append_trace(trace, row=1, col=1)

    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_flow(Frames, Nums, Nums_positiv, Nums_negativ, fps):
    logging.info("plot flow-curve")
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["<b>Flow</b>"],
        x_title="Time / s",
        y_title="J / 1/s",
    )
    maxx = -1    
    for i, _frames in Frames.items():
        nums = Nums[i]
        nums_positiv = Nums_positiv[i]
        nums_negativ = Nums_negativ[i]
        frames = _frames[:, 1]
        if not frames.size:
            continue

        if maxx < np.max(frames):
            maxx = np.max(frames)

        times = np.array(frames) / fps
        J = (nums - 1) / times
        J_positiv = (nums_positiv - 1) / times
        J_negativ = (nums_negativ - 1) / times
        J_negativ = J_negativ[J_negativ > 0]
        J_positiv = J_positiv[J_positiv > 0]
        trace = go.Scatter(
            x=np.hstack(([0, times[0]], times)),
            y=np.hstack(([0, 0], J)),
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            line=dict(width=3),
        )
        if J_positiv.any() and J_negativ.any():            
            trace_p = go.Scatter(
                x=np.hstack(([0, times[0]], times)),
                y=np.hstack(([0, 0], J_positiv)),
                mode="lines",
                showlegend=True,
                name=f"ID: {i}+",
                line=dict(width=3),
            )
            trace_n = go.Scatter(
                x=np.hstack(([0, times[0]], times)),
                y=np.hstack(([0, 0], J_negativ)),
                mode="lines",
                showlegend=True,
                name=f"ID: {i}-",
                line=dict(width=3),
            )
            fig.append_trace(trace_p, row=1, col=1)
            fig.append_trace(trace_n, row=1, col=1)

        fig.append_trace(trace, row=1, col=1)

    fig.update_layout(hovermode="x")
    fig.update_xaxes(
        range=[0, maxx / fps + 2],
    )
    # st.plotly_chart(fig, use_container_width=True)
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_time_distance(_frames, data, line, i, fps, num_peds, sample, group_index):
    frames_initial_speed_mean = 4 * fps  # Adrian2020a 4 s
    logging.info("plot time_distance curve")
    peds = _frames[:, 0].astype(int)
    num_peds = int(num_peds)
    sample = int(sample)
    frames = _frames[:, 1]
    peds = peds[:num_peds]
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[
            f"<b>Distance-Time for Transition {i} with {num_peds} pedestrians</b>"
        ],
        x_title="Distance to entrance / m",
        y_title="Time to entrance / s",
    )
    xstart = []
    ystart = []
    colors = []
    for p, toframe in zip(peds, frames):
        ff = data[np.logical_and(data[:, 1] <= toframe, data[:, 0] == p)]
        ped_group = data[data[:, 0] == p][0, group_index]
        if ped_group == 1:
            color = "blue"
        elif ped_group == 2:
            color = "red"
        elif ped_group == 3:
            color = "green"
        else:
            color = "black"

        speed = np.mean(ff[:frames_initial_speed_mean,  st.session_state.speed_index])
        sc = speed
        xx = []
        yy = []
        for (frame, x, y) in ff[::sample, 1:4]:
            pos = Point(x, y)
            dx = pos.distance(line)
            dt = (toframe - frame) / fps
            xx.append(dx)
            yy.append(dt)

        xstart.append(xx[0])
        ystart.append(yy[0])
        colors.append(sc)

        trace = go.Scatter(
            x=xx,
            y=yy,
            mode="lines",
            name=f"Agent: {p:.0f}",
            showlegend=False,
            line=dict(width=0.3, color=color),
        )
        fig.append_trace(trace, row=1, col=1)
    
    trace_start = go.Scatter(
        x=xstart,
        y=ystart,
        mode="markers",
        showlegend=False,
        name=f"Group: {ped_group:.0f}",
        marker=dict(
            size=5,
            color=colors,
            cmax=1,
            cmin=0,
            colorbar=dict(
                title="Initial speed / m/s"),
            colorscale="Jet",),
    )
    fig.append_trace(trace_start, row=1, col=1)

    return fig


def plot_peds_inside(frames, peds_inside, fps):
    logging.info("plot peds inside")
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["<b>Discharge curve</b>"],
        x_title="Time / s",
        y_title="Number of Pedestrians inside",
    )
    times = frames / fps
    trace = go.Scatter(
        x=times,
        y=peds_inside,
        mode="lines",
        name="<b>Discharge curve</b>",
        showlegend=False,
        line=dict(width=3, color="royalblue"),
    )
    fig.append_trace(trace, row=1, col=1)
    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_jam_lifetime(frames, lifetime, fps, title, ret, min_agents_jam):
    logging.info("plot jam_lifetime")
    max_avg = 0
    max_std = 0
    for i in range(ret.shape[0]):
        From = int(ret[i, 0])
        To = int(ret[i, 1])
        number_agents = np.mean(lifetime[From:To, 1])
        if number_agents > max_avg:
            max_avg = number_agents
            max_std = np.std(lifetime[From:To, 1])

    title = f"<b>Maximal Jam Lifetime: {title:.2f} [s]. Size of Jam: {max_avg:.0f} (+- {max_std:.0f})</b>"
    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Time / s",
        y_title="Number of Agents in Jam",
        subplot_titles=[title],
    )

    xx = np.zeros((len(frames), 2))
    xx[:, 0] = frames
    if lifetime.size:
        _, idx, _ = np.intersect1d(xx[:, 0], lifetime[:, 0], return_indices=True)
        xx[idx, 1] = lifetime[:, 1]

    times = xx[:, 0] / fps
    # the curve over all frames
    trace1 = go.Scatter(
        x=xx[0:, 0] / fps,
        y=xx[0:, 1],
        mode="lines",
        showlegend=False,
        name="Jam",
        # hoverinfo="skip",
        line=dict(width=3, color="royalblue"),
    )
    # horizontal line

    # jam curves from to
    for i in range(ret.shape[0]):
        From = int(ret[i, 0])
        To = int(ret[i, 1])
        trace2 = go.Scatter(
            x=lifetime[From:To, 0] / fps,
            y=lifetime[From:To, 1],
            mode="lines",
            showlegend=True,
            name=f"Lifetime: [{From/fps:.2f}, {To/fps:.2f}] ({(To-From)/fps:.2f})[s]",
            line=dict(width=3, color="red"),
            fill="tonexty",
        )
        trace3 = go.Scatter(
            x=[lifetime[From, 0] / fps, lifetime[To - 1, 0] / fps],
            y=[min_agents_jam, min_agents_jam],
            mode="lines",
            showlegend=False,
            name="Min agents in jam",
            # hoverinfo="skip",
            line=dict(width=3, dash="dash", color="grey"),
        )
        fig.append_trace(trace3, row=1, col=1)
        fig.append_trace(trace2, row=1, col=1)

    trace4 = go.Scatter(
        x=[xx[0, 0] / fps, xx[-1, 0] / fps],
        y=[min_agents_jam, min_agents_jam],
        mode="lines",
        showlegend=True,
        name="Min agents in jam",
        line=dict(width=3, dash="dash", color="grey"),
    )
    fig.append_trace(trace1, row=1, col=1)
    fig.append_trace(trace4, row=1, col=1)

    miny = np.min(xx[:, 1])
    maxy = np.max(xx[:, 1])
    minx = np.min(times)
    maxx = np.max(times)

    # fig.update_yaxes(
    #     range=[miny - 0.1, maxy + 0.1],
    # )
    # fig.update_xaxes(
    #     range=[minx - 0.1, maxx + 0.1],
    # )
    fig.update_layout(hovermode="x")
    return fig

def plot_jam_waiting_hist(waiting_time, fps, nbins):
    df = pd.DataFrame(
        waiting_time,
        columns=[
            "waiting",
        ],
    )
    if waiting_time.size:
        maxt = np.max(waiting_time)
    else:
        maxt = 0
    
    hist = px.histogram(
        df,
        x="waiting",
        marginal="rug",
        hover_data=df.columns,
        labels={"waiting": "Waiting time"},
        text_auto=True,
        nbins=nbins,
        title=f'<b>Maximal waiting time: {maxt:.2f} [s]</b>',
    )
    hist.update_layout(bargap=0.2)
    return hist


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_jam_lifetime_hist(chuncks, fps, nbins):
    chuncks = chuncks / fps
    df = pd.DataFrame(
        chuncks,
        columns=[
            "chuncks",
        ],
    )
    hist = px.histogram(
        df,
        x="chuncks",
        marginal="rug",
        hover_data=df.columns,
        labels={"chuncks": "Time"},
        text_auto=True,
        nbins=nbins,
        title='<b>Distribution of all Jam Lifetimes (in s)</b>',
    )
    hist.update_layout(bargap=0.2)
    return hist


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_RSET_hist(rset, nbins):

    rset = rset.flatten()
    df = pd.DataFrame(
        rset,
        columns=[
            "RSET",
        ],
    )
    hist = px.histogram(
        df,
        x="RSET",
        marginal="rug",
        hover_data=df.columns,
        labels={"RSET": "Time"},
        text_auto=True,
        nbins=nbins,
        title='<b>Distribution of RSET (in s)</b>',
    )
    hist.update_layout(bargap=0.2)
    return hist


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_timeserie(frames, t, fps, title, miny, maxy, liney=None):
    logging.info(f"plot timeseries: {title}")
    fig = make_subplots(rows=1, cols=1, x_title="Time / s", y_title=title)
    times = frames / fps
    trace = go.Scatter(
        x=times,
        y=t,
        name="",
        fill="tozeroy",
        mode="lines",
        showlegend=False,
        line=dict(width=3, color="royalblue"),
    )
    # plot line
    if liney is not None:
        trace1 = go.Scatter(
            x=[times[0], times[len(t)]],
            y=[liney, liney],
            name=f"Max Profile {title}",
            mode="lines", 
            showlegend=True,
            line=dict(width=3, dash="dash", color="gray"),
        )
        fig.append_trace(trace1, row=1, col=1)
        if liney > maxy:
            maxy = liney

    fig.append_trace(trace, row=1, col=1)
    fig.update_yaxes(
        range=[miny - 0.1, maxy + 0.1],
    )
    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_agent_xy(frames, X, Y, fps):
    logging.info("plot agent xy")
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        rows=1,
        cols=1,
        x_title="Time / s",
        subplot_titles=["<b>Trajectory of highlighted pedestrian</b>"],
    )
    times = frames / fps
    traceX = go.Scatter(
        x=times,
        y=X,
        mode="lines",
        showlegend=True,
        name="X",
        line=dict(width=3, color="firebrick"),
    )
    traceY = go.Scatter(
        x=times,
        y=Y,
        mode="lines",
        name="Y",
        showlegend=True,
        line=dict(width=3, color="royalblue"),
    )
    fig.add_trace(
        traceX,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        traceY,
        row=1,
        col=1,
        secondary_y=True,
    )
    # Set y-axes titles
    fig.update_yaxes(title_text="X", secondary_y=False)
    fig.update_yaxes(
        title_text="Y",
        secondary_y=True,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
  )
    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_agent_angle(pid, frames, angles, fps):
    logging.info("plot angle")
    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Time / s",
        y_title=r"Angle / Degree",
        subplot_titles=["<b>Orientation of highlighted pedestrian</b>"],
    )
    times = frames / fps
    trace = go.Scatter(
        x=times,
        y=angles,
        mode="lines",
        showlegend=False,
        fill="tozeroy",
        name=f"Agent: {pid:.0f}",
        line=dict(width=3, color="royalblue"),
    )
    fig.append_trace(trace, row=1, col=1)
    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_agent_speed(pid, frames, speed_agent, max_speed, fps):
    fig = make_subplots(
        rows=1,
        cols=1,
        x_title="Time / s",
        y_title="Speed / m/s",
        subplot_titles=["<b>Speed of highlighted pedestrian</b>"],
    )
    threshold = 0.5  # according to DIN19009-2
    logging.info(f"plot agent speed {pid}")
    m = np.copy(speed_agent)
    times = frames / fps
    tt = np.ones(len(speed_agent)) * threshold
    cc = np.isclose(m, tt, rtol=0.04)
    m[~cc] = None
    trace = go.Scatter(
        x=times,
        y=speed_agent,
        mode="lines",
        showlegend=False,
        name=f"Agent: {pid:.0f}",
        line=dict(width=3, color="royalblue"),
        stackgroup="one",
    )
    trace_threshold = go.Scatter(
        x=[times[0], times[-1]],
        y=[threshold, threshold],
        mode="lines",
        showlegend=True,
        name="Jam threshold",
        line=dict(width=4, dash="dash", color="gray"),
    )
    tracem = go.Scatter(
        x=times,
        y=m,
        mode="markers",
        showlegend=False,
        name="Jam speed",
        marker=dict(size=5, color="red"),
    )
    fig.append_trace(trace, row=1, col=1)
    fig.append_trace(trace_threshold, row=1, col=1)
    fig.append_trace(tracem, row=1, col=1)
    fig.update_yaxes(
        range=[0, max_speed + 0.01],
    )
    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_trajectories(
    data,
    special_ped,
    speed,
    geo_walls,
    transitions,
    min_x,
    max_x,
    min_y,
    max_y,
    choose_transitions,
    sample_trajectories,
):
    logging.info("plot trajectories")
    fig = make_subplots(rows=1, cols=1, subplot_titles=["<b>Trajectories</b>"])
    peds = np.unique(data[:, 0])
    for ped in peds:
        d = data[data[:, 0] == ped]
        trace_traj = go.Scatter(
            x=d[::sample_trajectories, 2],
            y=d[::sample_trajectories, 3],
            mode="lines",
            showlegend=False,
            name=f"Agent: {ped:0.0f}",
            line=dict(color="gray", width=0.3),
        )
        fig.append_trace(trace_traj, row=1, col=1)


    if special_ped > 0:
        s = data[data[:, 0] == special_ped]
        sc = speed / np.max(speed)
    
        trace_agent = go.Scatter(
            x=s[1::sample_trajectories, 2],
            y=s[1::sample_trajectories, 3],
            mode="markers",
            showlegend=False,
            name=f"Agent: {special_ped:0.0f}",
            marker=dict(size=5,
                        cmax=1,
                        cmin=0,
                        colorbar=dict(
                            title="Speed / m/s"),
                        color=sc,
                        colorscale="Jet"),
        )
        trace_agent_start = go.Scatter(
            x=[s[0, 2]],
            y=[s[0, 3]],
            mode="markers",
            showlegend=False,
            name=f"Start: {special_ped:0.0f}",
            marker=dict(size=10,
                        color="black",
                        ),
        )
        fig.append_trace(trace_agent, row=1, col=1)
        fig.append_trace(trace_agent_start, row=1, col=1)
    for gw in geo_walls.keys():
        trace_walls = go.Scatter(
            x=geo_walls[gw][:, 0],
            y=geo_walls[gw][:, 1],
            showlegend=False,
            mode="lines",
            line=dict(color="black", width=2),
        )
        fig.append_trace(trace_walls, row=1, col=1)

    if choose_transitions:
        for i, t in transitions.items():
            xm = np.sum(t[:, 0]) / 2
            ym = np.sum(t[:, 1]) / 2
            length = np.sqrt(np.diff(t[:, 0]) ** 2 + np.diff(t[:, 1]) ** 2)
            offset = 0.1 * length[0]
            logging.info(f"offset transition {offset}")
            trace_transitions = go.Scatter(
                x=t[:, 0],
                y=t[:, 1],
                showlegend=False,
                name=f"Transition: {i}",
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
            fig.append_trace(trace_transitions, row=1, col=1)
            fig.append_trace(trace_text, row=1, col=1)

    eps = 1
    fig.update_yaxes(
        range=[min_y - eps, max_y + eps],
    )
    fig.update_xaxes(
        range=[min_x - eps, max_x + eps],
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
  )
    return fig


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0], _geometry_wall[gw][:, 1], color="white", lw=2)



@st.cache(suppress_st_warning=True, hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_profile_and_geometry2(
        xbins,
        ybins,
        geometry_wall,
        xpos,
        ypos,
        lm,
        data,
        interpolation,
        label,
        title,
        vmin=None,
        vmax=None,
):
    """Plot profile + geometry for 3D data


    if vmin or vmax is None, extract values from <data>
    """
    logging.info("plot_profile and geometry 2")
    if vmin is None or vmax is None:
        vmin = np.min(data)
        vmax = np.max(data)

    if interpolation == "false":
        interpolation = False

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=([f'<b>{title}</b>']))

    heatmap = go.Heatmap(
        x=xbins,
        y=ybins,
        z=data,
        zmin=vmin,
        zmax=vmax,
        name=title,
        connectgaps=False,
        zsmooth=interpolation,
        hovertemplate="%{z:.2f}<br> x: %{x:.2f}<br> y: %{y:.2f}",
        colorbar=dict(
            title=f"{label}"),
        colorscale="Jet",
    )
    fig.add_trace(heatmap)
    #    Geometry walls
    for gw in geometry_wall.keys():
        line = go.Scatter(
            x=geometry_wall[gw][:, 0],
            y=geometry_wall[gw][:, 1],
            mode="lines",
            name="wall",
            showlegend=False,
            line=dict(
                width=3,
                color="white",
            )
        )
        fig.add_trace(line)

    # Measurement square
    if xpos is not None:
        fig.add_shape(
            x0=xpos-lm/2, x1=xpos+lm/2, y0=ypos-lm/2, y1=ypos+lm/2,
            xref='x', yref='y',
            line=dict(color="gray", width=4),
            type='rect',
        )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


@st.cache(
    suppress_st_warning=True, hash_funcs={matplotlib.figure.Figure: lambda _: None}
)
def plot_profile_and_geometry(
    geominX,
    geomaxX,
    geominY,
    geomaxY,
    geometry_wall,
    xpos,
    ypos,
    lm,
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
    logging.info("plot_profile and geometry")
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
    if xpos is not None:
        plot_square(ax, xpos, ypos, lm)

    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, rotation=90, labelpad=15, fontsize=15)
    return fig


def plot_square(ax, xpos, ypos, lm):
    x = [
        xpos - lm / 2,
        xpos - lm / 2,
        xpos + lm / 2,
        xpos + lm / 2,
        xpos - lm / 2,
    ]
    y = [
        ypos - lm / 2,
        ypos + lm / 2,
        ypos + lm / 2,
        ypos - lm / 2,
        ypos - lm / 2,
    ]
    ax.plot(x, y, color="gray", lw=2)


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_survival(Frames, fps):
    logging.info("plot survival function")
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["<b>Survival function of time gaps</b>"],
        x_title="Delta / s",
        y_title=r"P(t>Delta)",
    )
    for i, _frames in Frames.items():
        frames = _frames[:, 1]
        if frames.size < 2:
            continue

        times = np.array(frames) / fps
        y, dif = survival(times)
        trace = go.Scatter(
            x=dif,
            y=y,
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            line=dict(width=3),
        )
        fig.append_trace(trace, row=1, col=1)

    fig.update_yaxes(
        type="log",
        # range=[-1, 0]
    )
    fig.update_xaxes(
        type="log",
        # range=[-1, 1]
    )
    fig.update_layout(hovermode="x")
    return fig


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_vpdf(data):
    logging.info("plot speed pdf")
    speed = data[:, st.session_state.speed_index]
    speed = np.unique(speed)
    loc = speed.mean()
    scale = speed.std()
    pdf = stats.norm.pdf(speed, loc=loc, scale=scale)
    print(len(pdf), pdf)
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["<b>PDF Speed</b>"],
        x_title="Speed m / s",
        y_title=r"PDF",
    )

    trace = go.Scatter(
        x=speed,
        y=pdf,
        mode="lines",
        showlegend=False,
        line=dict(width=3),
        )

    fig.append_trace(trace, row=1, col=1)
    return fig
