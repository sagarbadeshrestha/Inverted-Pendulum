import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.integrate import odeint
from sympy import * # import all function names, whole library
from sympy.physics.vector import init_vprinting, dynamicsymbols, vpprint # for time non-static vector symbolic representation

init_vprinting() # Enhances display of dynamic symbols more beautifully, for normal ones uses init_printing()

# Symbolic representations of cart mass, point mass, length of rod, accn due to gravity, and force
m1, m2, l, g, F= symbols('m1 m2 l g F')

# Symbolic representation of state vectors x, x_dot, theta, theta_dot using dynamic symbols
x=dynamicsymbols('x')
x_dot= x.diff()
x_ddot= x.diff().diff()

theta=dynamicsymbols('θ')
theta_dot=theta.diff()
theta_ddot=theta.diff().diff()

# Writing down equations of motion of inverted pendulum
e1= (m1+m2)*x_ddot - m2*l*cos(theta)*theta_ddot + m2*l*sin(theta)*theta_dot**2 - F
e2= -cos(theta)*x_ddot + l*theta_ddot -g*sin(theta)

# We need to solve e1 and e2 to find x_ddot and theta_ddot since there are only two state variables each
# obtained from looking at the highest order of the eqations
result= solve([e1,e2],x_ddot,theta_ddot, dict=True)

# The obtained solutions were very long and complex, which can be simplified by using:
x_ddot_solved= simplify(result[0][x_ddot]) # result part is Accessing values from dictionary
theta_ddot_solved= simplify(result[0][theta_ddot]) 


### Generating state space equation i.e. xdot=Ax+Bu ### And write them in matrix form
states= Matrix([[x],
                [x_dot], 
                [theta], 
                [theta_dot]])
                  # basically x part of state equation

stateFxn= Matrix([[x_dot],
                  [x_ddot_solved], 
                  [theta_dot], 
                  [theta_ddot_solved]])
                  # basically xdot part of state equation

# Computing the jacobians
Jstate= stateFxn.jacobian(states) # jacobian of A
Jinput= stateFxn.jacobian([F])  # jacobian of B

# Adding constants to a dictionary to substitute them
consts={
    l: 1,
    g: 9.81,
    m1: 10,
    m2: 1
    }

# Substituting Constants into the jacobians
JstateSub= Jstate.subs(consts)
JinputSub= Jinput.subs(consts)

# Converting jacobians from symbolic expression to numerical
AmatFxn= lambdify([x, x_dot, theta, theta_dot, F],JstateSub)
BmatFxn= lambdify([x, x_dot, theta, theta_dot, F],JinputSub)

# Now providing equilibrium values to A and B matrices in order they were lambdified
# i.e. x, x_dot, theta ,theta_dot, F; they can be modified
x1, x2, x3 , x4, u= 0, 0, 0, 0, 0

A= AmatFxn(x1, x2, x3 , x4, u)
B= BmatFxn(x1, x2, x3 , x4, u)

# Creating output state matrices, assuming only angular components can be measured
C= np.array([
            [0,0,1,0],
            [0,0,0,1]
            ])

D= np.zeros((2,1))

# Defining a state space model
sysStateSpace= ct.ss(A,B,C,D)
print(sysStateSpace)

### Simulating open loop step response

startTime=0
endTime=20
samplesNum=1000
timeVector= np.linspace(startTime,endTime,samplesNum)
initState= np.array([0,0,0+0.3,0])
controlInput= np.ones(samplesNum)

OLsimulation=ct.forced_response(sysStateSpace,timeVector,controlInput,initState) # cannot be used for non-linearized system, if so then have to use odeint with a function that can return derivatives

states_data = OLsimulation.states.T 
combined_data = np.column_stack((timeVector, states_data))
np.save('openLoopSimulation.npy', combined_data)


"""
    These are the structure of outputs otained from forced response function
    
OLsimulation.time
OLsimulation.outputs
OLsimulation.states
OLsimulation.inputs
"""

# Create a figure with 4 subplots (2 rows, 2 column)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Plotting Cart and pendulum states
ax1.plot(OLsimulation.time, OLsimulation.states[0,:], 'b', linewidth=4, label='x')
ax1.set_xlabel('time', fontsize=16)
ax1.set_ylabel('Cart position', fontsize=16)
ax1.legend(fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.grid(True)

ax2.plot(OLsimulation.time, OLsimulation.states[1,:], 'b', linewidth=4, label='x')
ax2.set_xlabel('time', fontsize=16)
ax2.set_ylabel('Cart velocity', fontsize=16)
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid(True)

ax3.plot(OLsimulation.time, OLsimulation.states[2,:], 'b', linewidth=4, label='x')
ax3.set_xlabel('time', fontsize=16)
ax3.set_ylabel('Pendulum Position', fontsize=16)
ax3.legend(fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.grid(True)

ax4.plot(OLsimulation.time, OLsimulation.states[3,:], 'b', linewidth=4, label='x')
ax4.set_xlabel('time', fontsize=16)
ax4.set_ylabel('Pendulum velocity', fontsize=16)
ax4.legend(fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.grid(True)

plt.savefig('uncontrolledStates.png', dpi=600)

####      Designing LQR controller      #####

# Convert symbolic expressions to functions for simulation
x_ddot_func = lambdify([x, x_dot, theta, theta_dot, F], x_ddot_solved.subs(consts))
theta_ddot_func = lambdify([x, x_dot, theta, theta_dot, F], theta_ddot_solved.subs(consts))

### Selecting Q and R values and computing using LQR
#Q= 100*np.eye(4)
Q = np.diag([10, 1, 1000, 1])  
R= 1*np.array([[0.01]])

K, S, E = ct.lqr(sysStateSpace, Q, R) # where, K= gain matrix, S= Riccati solution, E= eigenvalues

# desired states
xd = np.array([0, 0, 0, 0])
ud=0

# creating function for closed loop system
def clStateSpace(state, t):
    # control input
    u= -np.dot(K, state - xd)
    
    x, xdot, theta, thetadot= state
    # Handle angle wrapping (keep theta between -pi and pi)
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    
    # Compute accelerations
    xddot= float(x_ddot_func(x,xdot,theta,thetadot,u))
    thetaddot= float(theta_ddot_func(x,xdot,theta,thetadot,u))
    
    return np.array([xdot, xddot, thetadot, thetaddot], dtype=float) #LHS of state space model

# Simulating closed loop system
clSolution= odeint(clStateSpace, initState, timeVector)

# Save the states
np.save('simulationControlled.npy', clSolution)

# Extract states
x_pos = clSolution[:, 0]
x_vel = clSolution[:, 1]
theta_pos = clSolution[:, 2]
theta_vel = clSolution[:, 3]

###  Calculating control effort to visualize it
controlEffort= np.zeros_like(timeVector)

for i, t in enumerate(timeVector):
    states= clSolution[i]
    # Handle angle wrapping (keep theta between -pi and pi)
    thetaWrapped= (states[2] + np.pi) % (2*np.pi) - np.pi
    # New states after wrapping
    statesWrapped= np.array([states[0], states[1], thetaWrapped, states[3]])
    # Control effort, i.e. force over time
    controlEffort[i]= -np.dot(K, statesWrapped - xd)

# Plot results
plt.figure(figsize=(12, 10))

# Cart position
plt.subplot(3, 2, 1)
plt.plot(timeVector, x_pos, 'b', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Cart Position (m)')
plt.grid(True)

# Cart velocity
plt.subplot(3, 2, 2)
plt.plot(timeVector, x_vel, 'r', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Cart Velocity (m/s)')
plt.grid(True)

# Pendulum angle
plt.subplot(3, 2, 3)
plt.plot(timeVector, theta_pos, 'g', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Angle (rad)')
plt.grid(True)

# Pendulum angular velocity
plt.subplot(3, 2, 4)
plt.plot(timeVector, theta_vel, 'm', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid(True)

# Control effort
plt.subplot(3, 2, 5)
plt.plot(timeVector, controlEffort, 'k', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.grid(True)

# Angle vs Angular velocity
plt.subplot(3, 2, 6)
plt.plot(theta_pos, theta_vel, 'c', linewidth=1)
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid(True)

plt.tight_layout()
plt.savefig('LQRoutputs.png', dpi=600)
plt.show()

