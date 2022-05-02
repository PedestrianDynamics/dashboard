[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/chraibi/jupedsim-dashboard/main/app.py)
[![Heroku](http://heroku-shields.herokuapp.com/jupedsim-dashboard)](https://jupedsim-dashboard.herokuapp.com/)


# JuPedSim-Dashboard

Show statistics and make plots extracted from [jpscore](https://github.com/jupedsim/jpscore)-simulations and [experimental data](https://ped.fz-juelich.de/db/).


<img width="1043" alt="Screen Shot 2022-03-25 at 20 47 58" src="https://user-images.githubusercontent.com/5772973/160191551-4e030612-e034-4c4c-af9c-38be83036e33.png">

## Use

This app can be used online: 

https://go.fzj.de/dashboard

or locally by running 

```bash
streamlit run app.py
```

To install the requirements use:

```bash
pip install -r requirements.txt
```

## Draw geometries 

To draw geometries on trajectory-plots, try to connect all lines, such that they form a closed polygon.
This is only necessary if you want to visualize the geometry with [jpsvis](https://www.jupedsim.org/).

You can download your drawing as an xml file according to jupedsim's [format](https://www.jupedsim.org/jpscore_geometry.html)

It is possible to draw: 
- Lines 
- Rectangles

Lines and rectangle can be rotated and scaled.

To try out:
https://share.streamlit.io/chraibi/jupedsim-dashboard/main/draw_geometry.py

See a Demo:

https://youtu.be/4xTqSbllCwg


**NOTE**: The script tries to exctract the unit of the data from the trajectory file. A good idea might be to convert the trajetories before using with this [script](https://github.com/JuPedSim/jpscore/blob/master/scripts/petrack2jpsvis.py)

or add a header with the unit information. For example a line starting with 
`#unit: cm`

## Profiles 
The density profile uses the speed of the pedestrians calculated by the Weidmann diagram.
Therefore, for jpscore-simulations, use the `optional_output`-option in the inifile:

```xml
<file location="bottleneck_traj.txt" />
    <optional_output speed="TRUE"/>
</trajectories>
```

## NT-curve 

NT-curves and flow at lines (`transitions` or `[area_L](https://www.jupedsim.org/jpsreport_inifile#measurement-area))
are calculates be testing for every agent, if croses the line.
Therefore, define in the geometry transitions, such that the pedestrian fully pass them. 

## Todo

- [x] Congestion (histogram)
- [X] split file Utilities into plot-functions and utilities 
- [X] Number of agents in geometry over time
- [X] Use forms to avoid running everything at once!
- [x] survival function
- [x] Choose between example data
- [X] Density time series in squares (l x l):
    -  sliders l, x, y
- [x] RSET maps
- [x] Accelerate calculation of N-T curves data
- [x] N_T curves bidirectional flow
- [ ] Density profiles slider: from and to frames
- [ ] Use steady state form this script https://github.com/JuPedSim/jpsreport/blob/develop/scripts/SteadyState.py 
  See also Cumulative sum algorithm https://github.com/BMClab/BMC/blob/master/functions/detect_cusum.py
- [x] Use density plots to detect the geometry
- [ ] LOS (green: d<0.8, yellow: d<1.6, red: d>1.6)
- [ ] show starting positions with colors
- [ ] Radio buttons to choose  between two different modes: 
  - Report mode 
  - Interactive mode
- [ ] time series: use the continuity equation to plot the flow as well
- [ ] Social distancing analysis (Qiancheng's paper)
- [ ] for faster plots check:
  - https://plotly.com/python/datashader/ 
  - https://datashader.org 
  - https://datashader.org/user_guide/Trajectories.html
- [ ] The developement of jam. 
  - Jam in front of the bottleneck
  - Jam is desolving 
  - More people are coming -> jam is going up
  - Dessolving 
  - going up again ...
  
- [ ] Queuing: Contraction of distances!

  
