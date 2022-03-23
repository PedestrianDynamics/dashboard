import lovely_logger as logging
import numpy as np
import streamlit as st
from pandas import read_csv
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
from shapely.geometry import LineString, Point


def get_fps(traj_file):
    fps = traj_file.split("#framerate:")[-1].split("\n")[0]
    try:
        fps = int(float(fps))
    except ValueError:
        logging.error(f"{fps} in header can not be converted to int")
        st.stop()

    return fps


def get_transitions(xml_doc):
    transitions = {}
    for _, t_elem in enumerate(xml_doc.getElementsByTagName('transition')):
        Id = t_elem.getAttribute('id')
        n_vertex = len(t_elem.getElementsByTagName('vertex'))
        vertex_array = np.zeros((n_vertex, 2))
        for v_num, _ in enumerate(t_elem.getElementsByTagName('vertex')):
            vertex_array[v_num, 0] = t_elem.getElementsByTagName('vertex')[v_num].attributes['px'].value
            vertex_array[v_num, 1] = t_elem.getElementsByTagName('vertex')[v_num].attributes['py'].value

        transitions[Id] = vertex_array

    return transitions


def passing_frame(ped_data: np.array, line: LineString, fps: int) -> int:
    """Return frame of first time ped enters the line buffer

    Enlarge the line by eps, a constant that is dependent on fps
    eps = 1/fps * v0, v0 = 1.3 m/s

    :param ped_data: trajectories of ped
    :param line: transition
    :param fps: fps

    :returns: frame of entrance. Return negative number if ped did not pass trans

    """
    eps = 1/fps * 1.3
    line_buffer = line.buffer(eps)
    for (frame, x, y) in ped_data[:, 1:4]:
        if Point(x, y).within(line_buffer):
            return frame

    return -1

def read_trajectory(input_file):
    data = read_csv(
        input_file, sep=r"\s+", dtype=np.float64, comment="#"
    ).values
    
    return data


def read_obstacle(xml_doc):
    # Initialization of a dictionary with obstacles
    return_dict = {}
    # read in obstacles and combine them into an array for polygon representation
    for o_num, o_elem in enumerate(xml_doc.getElementsByTagName('obstacle')):

        N_polygon = len(o_elem.getElementsByTagName('polygon'))
        if len(o_elem.getElementsByTagName('polygon')) == 1:
            pass
        else:
            points = np.zeros((0, 2))

        for p_num, p_elem in enumerate(o_elem.getElementsByTagName('polygon')):
            for v_num, v_elem in enumerate(p_elem.getElementsByTagName('vertex')):
                vertex_x = float(p_elem.getElementsByTagName('vertex')[v_num].attributes['px'].value)
                vertex_y = float(p_elem.getElementsByTagName('vertex')[v_num].attributes['py'].value)
                points = np.vstack([points , [vertex_x, vertex_y]])

        points = np.unique(points, axis=0)
        x = points[:, 0]
        y = points[:, 1]
        n = len(points)
        center_point = [np.sum(x)/n, np.sum(y)/n]
        angles = np.arctan2(x-center_point[0],y-center_point[1])
        ##sorting the points:
        sort_tups = sorted([(i,j,k) for i,j,k in zip(x,y,angles)], key = lambda t: t[2])
        return_dict[o_num] = np.array(sort_tups)[:,0:2]

    return return_dict


def read_subroom_walls(xml_doc):
    dict_polynom_wall = {}
    n_wall = 0
    for _, s_elem in enumerate(xml_doc.getElementsByTagName('subroom')):
        logging.info(f"Get subroom: {s_elem.getAttribute('id')}")
        for _, p_elem in enumerate(s_elem.getElementsByTagName('polygon')):
            if p_elem.getAttribute('caption') == "wall":
                
                n_wall = n_wall + 1                
                n_vertex = len(p_elem.getElementsByTagName('vertex'))
                vertex_array = np.zeros((n_vertex, 2))

                for v_num, _ in enumerate(p_elem.getElementsByTagName('vertex')):
                    vertex_array[v_num, 0] = p_elem.getElementsByTagName('vertex')[v_num].attributes['px'].value
                    vertex_array[v_num, 1] = p_elem.getElementsByTagName('vertex')[v_num].attributes['py'].value

                dict_polynom_wall[n_wall] = vertex_array

    return dict_polynom_wall


def geo_limits(geo_xml):
    geometry_wall = read_subroom_walls(geo_xml)
    geominX = 1000
    geomaxX = -1000
    geominY = 1000
    geomaxY = -1000
    Xmin = []
    Ymin = []
    Xmax = []
    Ymax = []
    for k, _ in geometry_wall.items():
        Xmin.append(np.min(geometry_wall[k][:, 0]))
        Ymin.append(np.min(geometry_wall[k][:, 1]))
        Xmax.append(np.max(geometry_wall[k][:, 0]))
        Ymax.append(np.max(geometry_wall[k][:, 1]))

    geominX = np.min(Xmin)
    geomaxX = np.max(Xmax)
    geominY = np.min(Ymin)
    geomaxY = np.max(Ymax)
    return geominX, geomaxX, geominY, geomaxY


# def plot_peds(ax):
#     for i in ids:
#         d = data[data['i'] == i]
#         x = d['x']
#         y = d['y']
#         ax.plot(x, y, 'k', lw=0.05)

def get_geometry_file(traj_file):
    return traj_file.split("geometry:")[-1].split("\n")[0]


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0], _geometry_wall[gw][:, 1], color='white', lw=1)
