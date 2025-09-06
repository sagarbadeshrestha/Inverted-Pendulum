import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Load simulation data
data = np.load('simulationData.npy')

l = 1.0  # Pendulum length fixed

# 5000 values stored along cols x, xDot, theta, thetaDot i.e. 2d array
x = data[:, 0]
theta = data[:, 2]

fig, ax = plt.subplots(figsize=(8, 6))

# Define a scale factor to exaggerate cart movement for visibility
scale = 2.0

# Adjusted axis limits with margin for better motion visibility
ax.set_xlim((np.min(x)*scale - 2*l, np.max(x)*scale + 2*l)) 
ax.set_ylim((-1.8*l, 1.5*l))
ax.set_aspect('equal')  # ensures same scaling along x and y axis to prevent skewing
ax.grid()

# Cart dimensions
cart_width = 0.4
cart_height = 0.2
wheel_radius = 0.07

##  The 0,0 coordinates below are just placeholders which will get updated later
# Cart rectangle
cart = patches.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
ax.add_patch(cart)

# Two wheels on bottom corners of the cart
wheel_left = patches.Circle((0, 0), wheel_radius, fc='grey')
wheel_right = patches.Circle((0, 0), wheel_radius, fc='grey')
ax.add_patch(wheel_left)
ax.add_patch(wheel_right)

pendulum_line, = ax.plot([], [], lw=3, c='red')

bob_radius = 0.05
pendulum_bob = patches.Circle((0, 0), bob_radius, fc='red', ec='black')
ax.add_patch(pendulum_bob)

# Ground line below wheels (wheels sit on y=0, ground just below)
ground_y = -wheel_radius * 1.1
ax.plot([-10, 10], [ground_y, ground_y], 'k-', lw=2)    # k- means black solid line

def init():
    cart.set_xy((-cart_width/2, 0))
    wheel_left.center = (0, 0)
    wheel_right.center = (0, 0)
    
    pendulum_line.set_data([], [])
    pendulum_bob.center = (0, 0)
    return cart, wheel_left, wheel_right, pendulum_line, pendulum_bob

def update(frame):
    # Scale cart position for visibility
    cart_x = x[frame] * scale
    
    cart.set_xy((cart_x - cart_width/2, 0))
    
    # Position wheels relative to cart rectangle
    wheel_left.center = (cart_x - cart_width/3, 0)
    wheel_right.center = (cart_x + cart_width/3, 0)
    
    # Adjust angle for pendulum below the cart
    angle = theta[frame] - np.pi
    
    pivot_x = cart_x
    pivot_y = cart_height
    
    # Resolving pendulum length along x and y component
    bob_x = pivot_x + l * np.sin(angle)
    bob_y = pivot_y - l * np.cos(angle)
    
    pendulum_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
    pendulum_bob.center = (bob_x, bob_y)
    
    #length = np.sqrt((bob_x - pivot_x)**2 + (bob_y - pivot_y)**2)
    #print(f"Frame {frame}: Pendulum length = {length:.4f}")

    return cart, wheel_left, wheel_right, pendulum_line, pendulum_bob

ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=20)

plt.title('Inverted Pendulum on Cart Simulation')
plt.show()
