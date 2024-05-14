import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from math import tau 
from scipy.integrate import quad_vec  
from tqdm import tqdm 
import matplotlib.animation as animation 

# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

img = cv2.imread("nepal.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img_gray, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("edges_image.jpg", edges)

# find the contours in the image
# contours are retrieved as a list of arrays
ret, thresh = cv2.threshold(img_gray, 127, 255, 0) 
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
contours = np.array(contours[1])

# print("Contours : ", contours)

# split the co-ordinate points of the contour
x_list, y_list = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)

x_list = x_list - np.mean(x_list)
y_list = y_list - np.mean(y_list)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_list, y_list)

xlim_data = plt.xlim() 
ylim_data = plt.ylim()

# print("xlim", xlim_data)
# print("ylim", ylim_data)

plt.show()

# time data from 0 to 2*PI
t_list = np.linspace(0, tau, len(x_list))

order = 100 

print("generating coefficients ...")
c = []
pbar = tqdm(total=(order*2+1))
for n in range(-order, order+1):
    coef = 1/tau*quad_vec(lambda t: f(t, t_list, x_list, y_list)*np.exp(-n*t*1j), 0, tau, limit=100, full_output=1)[0]
    c.append(coef)
    pbar.update(1)
pbar.close()
print("completed generating coefficients.")

c = np.array(c)

np.save("coeff.npy", c)

# to store the points of last circle of epicycle which draws the required figure
draw_x, draw_y = [], []

# make figure for animation
fig, ax = plt.subplots()


circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order+1)]
circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-order, order+1)]
drawing, = ax.plot([], [], 'k-', linewidth=2)

orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

# ensure that figure does not get cropped
ax.set_xlim(xlim_data[0]-200, xlim_data[1]+200)
ax.set_ylim(ylim_data[0]-200, ylim_data[1]+200)
ax.set_axis_off()
ax.set_aspect('equal')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Naieem Islam'), bitrate=1800)

print("compiling animation ...")
frames = 300
pbar = tqdm(total=frames)


def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order+1):
        new_coeffs.extend([coeffs[order+i],coeffs[order-i]])
    return np.array(new_coeffs)

# make frame at time t
def make_frame(i, time, coeffs):

    global pbar
    t = time[i]

    #for making rotation of circle
    exp_term = np.array([np.exp(n*t*1j) for n in range(-order, order+1)])
    coeffs = sort_coeff(coeffs*exp_term) 

    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    center_x, center_y = 0, 0

    # make all epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        # calculate radius of current circle
        r = np.linalg.norm([x_coeff, y_coeff])

        # draw circle with given radius at given center points of circle
        # x = center_x + r * cos(theta), y = center_y + r * sin(theta)
        theta = np.linspace(0, tau, num=50)
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)

        # draw a line to indicate the direction of circle
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)

        # calculate center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff
    
    draw_x.append(center_x)
    draw_y.append(center_y)
    drawing.set_data(draw_x, draw_y)
    orig_drawing.set_data(x_list, y_list)

    pbar.update(1)

# for animation
time = np.linspace(0, tau, num=frames)
anim = animation.FuncAnimation(fig, make_frame, frames=frames, fargs=(time, c),interval=5)
anim.save('nepal.mp4', writer=writer)
pbar.close()
print("completed: nepal.mp4")