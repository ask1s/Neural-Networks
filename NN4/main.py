import math
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import cv2
from PIL import ImageTk, Image
from sklearn.cluster import KMeans
from sklearn import datasets

iris_df = datasets.load_iris()
model = KMeans(n_clusters=3)
qq = -1
ww = -2
first = True
ones = True

def color_code(targets):
    code = {
        0: 'red',
        1: 'green',
        2: 'blue'
    }

    return [code[target] for target in targets]

#input visualization
root = Tk()
root.title('KMeans')
root.geometry("600x400")
bg = ImageTk.PhotoImage(file="iris.png")
canvas= Canvas(root, width=600, height=400)
canvas.pack(fill="both", expand=True)
canvas.create_image(0,0, image=bg, anchor="nw")

def click_1():
    butn3.destroy()
    global qq
    global ww
    global first
    if first == True:
        first = False
        qq = 0
    else:
        if qq != 0:
            ww = 0
        root.destroy()

def click_2():
    butn4.destroy()
    global qq
    global ww
    global first
    if first == True:
        first = False
        qq = 1
    else:
        if qq != 1:
            ww = 1
        root.destroy()
def click_3():
    butn1.destroy()
    global qq
    global ww
    global first
    if first == True:
        first = False
        qq = 2
    else:
        if qq !=2:
            ww = 2
        root.destroy()

def click_4():
    butn2.destroy()
    global qq
    global ww
    global first
    if first == True:
        first = False
        qq = 3
    else:
        if qq != 3:
            ww = 3
        root.destroy()


canvas.create_text(300,40, fill="white",font=("Italic bold",20), text="Choose the 2 classes to be as input")
butn1 = Button(root, text="Petal Length", font=("Italic bold", 20), width=15, fg="#000000", command=click_3)
butn2 = Button(root, text="Petal Width", font=("Italic bold", 20), width=15, fg="#000000", command=click_4)
butn3 = Button(root, text="Sepal Length", font=("Italic bold", 20), width=15, fg="#000000", command=click_1)
butn4 = Button(root, text="Sepal Width", font=("Italic bold", 20), width=15, fg="#000000", command=click_2)

b1 = canvas.create_window(50,100, anchor="nw", window=butn1)
b2 = canvas.create_window(300,100, anchor="nw", window=butn2)
b3 = canvas.create_window(50,250, anchor="nw", window=butn3)
b4 = canvas.create_window(300,250, anchor="nw", window=butn4)

root.mainloop()

patterns = []
classes = []

if qq == 0:
    tt1 = "Sepal Length"
elif qq == 1:
    tt1 = "Sepal Width"
elif qq == 2:
    tt1 = "Petal Length"
elif qq == 3:
    tt1 = "Petal Width"

if ww == 0:
    tt2 = "Sepal Length"
elif ww == 1:
    tt2 = "Sepal Width"
elif ww == 2:
    tt2 = "Petal Length"
elif ww == 3:
    tt2 = "Petal Width"

filename = 'Iris_data.txt'
file = open(filename,'r')

for line in file.readlines():
    row = line.strip().split(',')
    patterns.append([row[int(qq)], row[int(ww)]])
    classes.append(row[4])

patterns = np.asarray(patterns,dtype=np.float32)


def mapunits(input_len, size='small'):

    heuristic_map_units = 5 * input_len ** 0.54321

    if size == 'big':
        heuristic_map_units = 4 * (heuristic_map_units)
    else:
        heuristic_map_units = 0.25 * (heuristic_map_units)

    return heuristic_map_units


map_units = mapunits(len(patterns), size='big')

model.fit(iris_df.data)

input_dimensions = 2
map_width = 8
map_height = 8
MAP = np.random.uniform(size=(map_height, map_width, input_dimensions))
prev_MAP = np.zeros((map_height, map_width, input_dimensions))

radius0 = max(map_width, map_height) / 2 #sigma
learning_rate0 = 0.1

epochs = 400
radius = radius0
learning_rate = learning_rate0

convergence = [1]
e = 0.001
flag = 0

for epoch in range(epochs):

    shuffle = np.random.randint(len(patterns), size=len(patterns))
    for i in range(len(patterns)):

        J = np.linalg.norm(MAP - prev_MAP)

        if J <= e:
            flag = 1
            break
        else:
            pattern = patterns[shuffle[i]]
            pattern_ary = np.tile(pattern, (map_height, map_width, 1))
            Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)

            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)

            prev_MAP = np.copy(MAP)

            for i in range(map_height):
                for j in range(map_width):
                    distance = np.linalg.norm([i - BMU[0], j - BMU[1]])
                    if distance <= radius:
                        MAP[i][j] = MAP[i][j] + learning_rate * (pattern - MAP[i][j]) #theta
            learning_rate = learning_rate0 * (1 - (epoch / epochs))
            radius = radius0 * math.exp(-epoch / epochs)


    if J < min(convergence):
        BMU = np.zeros([2], dtype=np.int32)
        result_map = np.zeros([map_height, map_width, 3], dtype=np.float32)
        n = 0
        for pattern in patterns:

            pattern_ary = np.tile(pattern, (map_height, map_width, 1))
            Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)

            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)

            x = BMU[0]
            y = BMU[1]

            if classes[n] == 'Iris-setosa':
                if result_map[x][y][0] <= 0.5:
                    result_map[x][y] += np.asarray([0.5, 0, 0])
            elif classes[n] == 'Iris-virginica':
                if result_map[x][y][1] <= 0.5:
                    result_map[x][y] += np.asarray([0, 0.5, 0])
            elif classes[n] == 'Iris-versicolor':
                if result_map[x][y][2] <= 0.5:
                    result_map[x][y] += np.asarray([0, 0, 0.5])
            if n == 150:
                n = 0
                break
            else:
                n += 1
        if ones == True:
            result_map = np.flip(result_map, 1)
            result_map = np.rot90(result_map)
            result_map = np.rot90(result_map)
            result_map = np.rot90(result_map)
            ones = False

        res = cv2.imwrite('Result Image.png', result_map)
        all_predictions = model.predict(iris_df.data)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].scatter(patterns[:,0], patterns[:,1], c=color_code(all_predictions))
        axs[0].set_xlabel(tt1)
        axs[0].set_ylabel(tt2)

        axs[1].imshow(result_map)
        axs[1].set_xlabel(tt1)
        axs[1].set_ylabel(tt2)
        plt.title(f'Epoch: {epoch}')
        plt.show()

    convergence.append(J)

    if flag == 1:
        break

#print('Final error: ' + str(J))
#print('Neighbourhood radius: ' + str(radius))

print("Red = Iris-Setosa")
print("Blue = Iris-Virginica")
print("Green = Iris-Versicolor")

