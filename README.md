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
- [ ] Density profiles slider: from and to frames
- [ ] N_T curves bidirectional flow
- [ ] Radio buttons to choose  between two different modes: Report vs Interactive
- [ ] Social distancing analysis (Qiancheng's paper)
- [ ] for faster plots check:
  - https://plotly.com/python/datashader/ 
  - https://datashader.org 
  - https://datashader.org/user_guide/Trajectories.html

