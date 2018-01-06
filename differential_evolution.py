# coding: utf-8

# # Cost Function - Polygons

import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import random
import sys

nr_poly = 10
nr_vertex = 3
imagename = "mona_lisa.jpg"

if len(sys.argv) > 1:
    imagename = str(sys.argv[1])

img = Image.open(imagename).convert("RGB")
image = np.array(img)
size = tuple(reversed(image.shape[:2]))  # W and H of image

popsize = 50  # Population size, must be >= 4
mutate = 0.5  # Mutation factor [0,2]
recombination = 0.7  # Recombination rate [0,1]
maxiter = 5  # Max number of generations (maxiter)


# Calculates difference between two images. It done subtracting one image from another.
# img1 and img2 are images in numpy array format.
def abs_error(img1, img2):
    return np.sum(np.abs(img1.astype("int32") - img2.astype("int32")))


def square_error(img1, img2):
    return np.sum(np.square(img1.astype("int64") - img2.astype("int64")))

# Generates image from polygons
# size - image height and width
# polygons - list of polygons, in shape [[R,G,B,A,X0,Y0, ..., Xn, Yn],

def gen_image_poly(size, polygons):
    image = Image.new("RGB", size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(image, mode="RGBA")

    nr_vertex = (len(polygons[0]) - 4) // 2  # TODO make it better

    for pol in polygons:
        fill = np.concatenate([pol[:3], [pol[3]]])  # Scale fill between 0 to 100.
        fill = tuple(map(int, fill))  # Make R,G,B to ints and leave alpha as it is
        points = np.reshape(pol[4:], newshape=(nr_vertex, 2))  # TODO number of vertices must be known?
        points = tuple(map(tuple, points))
        draw.polygon(points, fill=fill)

    img_mat = np.array(image)

    del image
    del draw

    return img_mat


# image
# np.asarray(image).shape


# In[38]:


# Cost function for using polygons
# orig_mat - original image given as matrix
# nr_poly - number of polygons used
# params - information about polygons in 1-D array (R,G,B,A,x0,y0,...,xn,yn)

def cost_poly(params, orig_mat, nr_poly):
    polygons = np.split(params, nr_poly)
    size = tuple(reversed(image.shape[:2]))  # W and H of image

    new_mat = gen_image_poly(size=size, polygons=polygons)

    return square_error(orig_mat, new_mat)


#mona_lisa = Image.open("mona_lisa.jpg")

#darwin = Image.open("darwin.PNG").convert("RGB")

#square = Image.open("square.png").convert("RGB")


# In[46]:


#orig_square = np.array(square)

# ## Try differential evolution


# Return bounds based on image size and number of polygons and vertices
# nr_poly - number of polygons
# nr_vertex - number of vertices for each polygon
# size - (width, height)
# rgb_range - default RGB range
# alpha_range - default alpha range
def get_bounds(nr_poly, nr_vertex, image, rgb_range=None, alpha_range=None):
    if alpha_range is None:
        alpha_range = [0, 255]
    if rgb_range is None:
        rgb_range = [0, 255]
    size = tuple(reversed(image.shape[:2]))
    w_range = [1, size[0]]
    h_range = [1, size[1]]

    polygon = 3 * [rgb_range] + [alpha_range] + nr_vertex * [w_range, h_range]
    bounds = nr_poly * polygon

    return bounds

bounds = get_bounds(nr_poly, nr_vertex, image)

# inspired by https://nathanrooy.github.io/posts/2017-08-27/simple-differential-evolution-with-python/

def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):
        polygon = []
        for j in range(len(vec[i])):
            # variable exceedes the minimum boundary
            if vec[i][j] < bounds[j][0]:
                polygon.append(bounds[j][0])

            # variable exceedes the maximum boundary
            if vec[i][j] > bounds[j][1]:
                polygon.append(bounds[j][1])

            if bounds[j][0] <= vec[i][j] <= bounds[j][1]:
                polygon.append(vec[i][j])
        vec_new.append(polygon)
    return vec_new


def differential_evolution(cost_func, bounds, popsize, mutate, recombination, maxiter, orig_mat, nr_poly):
    population = []
    for i in range(0, popsize):
        indv = []
        for j in range(nr_poly):
            polygons = []
            # todo - 14 should not be hardcoded
            for k in range(14):
                polygons.append(random.randint(bounds[k][0], bounds[k][1]))
            indv.append(polygons)
        population.append(indv)

    for i in range(1, maxiter + 1):
        gen_scores = []  # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = [*range(0, popsize)]
            canidates.remove(j)
            random_index = random.sample(canidates, 3)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]  # target individual

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = []
            for m in range(0, len(x_2)):
                eldiff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2[m], x_3[m])]
                x_diff.append(eldiff)

            v_donor = []
            for v in range(0, len(x_diff)):
                donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1[v], x_diff[v])]
                v_donor.append(donor)
            v_donor = ensure_bounds(v_donor, bounds)

            v_trial = []
            # cycle through each variable in our target vector
            for k in range(len(x_t)):
                crossover = random.random()
                # recombination occurs when crossover <= recombination rate
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                # recombination did not occur
                else:
                    v_trial.append(x_t[k])
            score_trial = cost_func(np.array(v_trial).flatten(), orig_mat, nr_poly)
            score_target = cost_func(np.array(x_t).flatten(), orig_mat, nr_poly)

            if score_trial < score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)

            else:
                gen_scores.append(score_target)
        # gen_avg = sum(gen_scores) / popsize  # current generation avg. fitness
        # gen_best = min(gen_scores)  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]  # solution of best individual
        #new_img = gen_image_poly(size, gen_sol)
        #print(i)
        #plt.imshow(new_img)
        #plt.show()

    return gen_sol


cost_func = cost_poly  # Cost functioni sees võrreldakse originaalpildig


# --- RUN ----------------------------------------------------------------------+

a = differential_evolution(cost_func, bounds, popsize, mutate, recombination, maxiter, image, nr_poly)

from datetime import datetime
date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

outputfile = "DE-"+str(date)+"-"+str(imagename.split(".")[0])+"-vert"+str(nr_vertex)+"-iter"+str(maxiter)+"-poly"+str(nr_poly)+".png"
outputimage = gen_image_poly(size, a)
plt.imsave(outputfile,outputimage)

#print(a)
