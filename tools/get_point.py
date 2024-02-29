import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def onclick(event):
    ix, iy = event.xdata, event.ydata
    print(f'x = {int(ix)}, y = {int(iy)}')

fig = plt.figure()
img = mpimg.imread('vis/init.png') # Replace 'your_image_path_here.jpg' with the path to your image
imgplot = plt.imshow(img)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()