# Human-MRS bounding overwatch with Bayesian optimization online learning human trust dynamics

This project includes two rospkgs: `autonomous_bounding_overwatch ` and ` trust_motion_plannar `. (Ubuntu 18.0 ROS Melodic)

## 1. autonomous_bounding_overwatch
It aims to verify that the Bayesian optimization can generate the posterior distribution approximating to the assigned simulated human hyperparameters. No human operated robot and only one platoon (3 robots) advances intermittently. The program relies on the assumed ground-truth value of trust model parameters and the autonomous robots' real-time sensing to generate trust feedback. We generate the simulated data with x^k_i = beta_true * z_sensing + noise, y^k_i = x^k_i - x^k-1_i + noise. 

#### Simulation without runing Gazebo simulator
```
roslaunch autonomous_bounding_overwatch simulated_human2_bo.launch
```
Note: the printed alpha is the array of the probability of each path to be selected with the acquisition function. The printed next line is the beta's posterior.

#### Simulation runing Gazebo simulator
```
roslaunch autonomous_bounding_overwatch multi_husky_terrain_gazebo.launch
roslaunch autonomous_bounding_overwatch autonomous_motion.launch
```

Note1: we do not rely on a robot to generate the trust feedback data. The program relies on the assumed ground truth value of trust model parameters and the autonomous robots' real-time sensing to generate trust feedback. 

Note2: the output `which path: #` is the selected path for the data collection after each iteration of Bayesian optimization. `#` ranges from 0 ($\rho_1$) to 4 ($\rho_5$) for paths 1 to 5.

## 2. trust_motion_plannar
It aims to verify that the Bayesian optimization can generate the posterior distribution of a human participant's trust model. Participant operated robot and one platoon (3 robots) advance intermittently.

#### BOED is also compared with the stanard experimental design:

Each experimental design has 6 trials. The first 5 trials provide training data, while 6th trial provides testing data in the goodness of fit. 
Both experimental designs have the same operating procedures in every trial. However, the standard experimental design is sequentially assigned paths $rho_1$ - $rho_5$, and $rho_6$. BOED uses the acquisition function to recommend a preferable path based on the up-to-date posterior distribution of model parameters at the end of every trial.

#### Operation procedures in one trial:

(1) A discrete path for the human-MRS is generated according to the decision field theory based acquisition function;

(2) the three-robot formed subteam autonomously navigates from the current cell to a temporary destination in the neighboring cell along the selected discrete path. The team then stops; 

(3) The terminal prints out that `the autonomous robots reached the temporal destination` and `human should provide trust changes on the HCI`. Then, the human operator should evaluate the trust change on HCI and click `publish trust` to store the trust value. Next, the human operator directly `manipulates the manned ground robot to bound to the autonomous robots` along the same discrete path. Note we are using a keyboard (i,j,k,l keys) to teleoperate; Click the terminal that runs teleoperation codes, then use the i,j,k,l keys to teleoperate the manual controlled robots in Gazebo. 

(4) meanwhile, the autonomous robots overwatch the surrounding environment;

(5) the human operator provides trust change in each autonomous robots by referring to the recorded traversability and visibility information of autonomous robots; 

(6) repeats steps (2) - (5) until all the ground robots reach the ultimate destination.


#### Simulation runing with standard experimental design
```
roslaunch trust_motion_plannar multi_husky_terrain_gazebo.launch
roslaunch trust_motion_plannar keyboard_teleop_bi.launch
```

Note: "No inter-relation" and "inter-relation" are in the same codes "roslaunch trust_motion_plannar keyboard_teleop_bi.launch". They are estimated simultaneously in one participant's operation.


#### Simulation runing with Bayesian optimization
```
roslaunch trust_motion_plannar multi_husky_terrain_gazebo.launch
roslaunch trust_motion_plannar keyboard_teleop2.launch
```


#### Data postprocessing: participants 1 - 16 are for standard experimental design, participants 17 - 32 are for Bayesian optimization.

Code Directory: ~/catkin_ws_hz/src/trust_motion_plannar/node

##### Forecasting accuracy: for each participant (1 - 16)

1. Replace the files in `~/catkin_ws_hz/src/trust_motion_plannar/node/data/` (lab computer) with the files in `workspace_data_human_subjsect_test/participant#/` (repository)

2. Open the `summary.txt` file in step 1, copy the values of `model2 beta: ...` (e.g., -0.008494066272159142, 1.0814993693365587, 0.490698474426846, 0.676094564920843, 0.05452330960200013 in participant1) and paste the values into `line 220` of `~/catkin_ws_hz/src/trust_motion_plannar/node/prediction.py`

3. Run the following code:

```
python prediction.py  # generate the prediction accuracy related data for a participant
```
4. The printed output is the forecasting accracy. The plots are the current participant's traversability and visibility along the path $rho_1$ at the 6th trial of the standard experiment. 

##### BIC: wilcoxon signed-rank test
1. See the sheet "goodness-of-fit" in the file `workspace_data_human_subjsect_test/human_subject_test.xlsx' (repository). 

2. In the sheet, see the organized BIC values from `summary.txt` of every participant (1 - 16)

3. Find an online wilcoxon signed-rank test calculator to analyze the p-value for the two groups of BIC value (e.g., https://www.socscistatistics.com/tests/signedranks/default2.aspx)

##### six-metric: Kruskal–Wallis one-way analysis of variance (ANOVA) test
1. See the sheet "performance", "workload", "usability" and "situational awareness" in the file `workspace_data_human_subjsect_test/human_subject_test.xlsx' (repository). 

2. In the sheet, see the metric values of each group's participants.

3. Find an online Kruskal–Wallis one-way ANOVA test calculator to analyze the p-value for the two groups of metric's value (e.g., https://www.socscistatistics.com/tests/kruskal/default.aspx)

##### BIC (plot 1 - 16 participants together) & six-metric (plot 1 - 32 participants together) 
The BIC and six-metric are plot with the following code (~/catkin_ws_hz/src/trust_motion_plannar/node/boxplot.py):
```
python boxplot.py  # generate the human subject test related data for all participants
```
 
