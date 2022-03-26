[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/chraibi/jupedsim-dashboard/main/app.py)
[![Heroku](http://heroku-shields.herokuapp.com/jupedsim-dashboard)](https://jupedsim-dashboard.herokuapp.com/)


# JuPedSim-Dashboard

Show statistics and make plots extracted from [jpscore](https://github.com/jupedsim/jpscore)-simulations.


<img width="1043" alt="Screen Shot 2022-03-25 at 20 47 58" src="https://user-images.githubusercontent.com/5772973/160191551-4e030612-e034-4c4c-af9c-38be83036e33.png">


## Profiles 
The density profile uses the speed of the pedestrians calculated by the Weidmann diagram.
Therefore, use the `optional_output`-option in the inifile:

```xml
<file location="bottleneck_traj.txt" />
    <optional_output speed="TRUE"/>
</trajectories>
```

## NT-curve 

Define in the geometry transitions, such that the pedestrian fully pass them. 


## Todo

- [ ] Congestion (histogram)
- [ ] split file Utilities into plot-functions and utilities 
- [ ] Number of agents in geometry over time
- [ ] 
