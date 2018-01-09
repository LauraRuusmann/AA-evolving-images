# Given file contains scripts for hill-climbing using polygon cost function

import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from random import randint, choice, random
import os

# Square error between images
def square_error(img1, img2):
    return np.sum(np.square(img1.astype("int64") - img2.astype("int64")))
    
    
# Generates image from polygons
    #size - image height and width
    #polygons - list of polygons, in shape [[R,G,B,A,X0,Y0, ..., Xn, Yn], 
def gen_image_poly(size, polygons):

    image = Image.new("RGB", size, (255,255,255,0)) 
    draw = ImageDraw.Draw(image, mode="RGBA")

    nr_vertex = (len(polygons[0]) - 4)//2 # TODO make it better

    for pol in polygons:
        fill = np.concatenate([pol[:3], [pol[3]]]) # Scale fill between 0 to 100.
        fill = tuple(map(int, fill))  # Make R,G,B to ints and leave alpha as it is
        points = np.reshape(pol[4:], newshape=(nr_vertex,2))  # TODO number of vertices must be known?
        points = tuple(map(tuple, points))
        draw.polygon(points, fill= fill)
    
    img_mat = np.array(image)

    del image
    del draw

    return img_mat


# Cost function for using polygons
    # orig_mat - original image given as matrix
    # nr_poly - number of polygons used
    # params - information about polygons in 1-D array (R,G,B,A,x0,y0,...,xn,yn)
def cost_poly(params, orig_mat, nr_poly):
    
    polygons = np.split(params, nr_poly)
    size = tuple(reversed(orig_mat.shape[:2])) # W and H of image
    
    new_mat = gen_image_poly(size = size, polygons = polygons)
        
    return square_error(orig_mat, new_mat)    


# Return bounds based on image size and number of polygons and vertices
    # nr_poly - number of polygons
    # nr_vertex - number of vertices for each polygon
    # size - (width, height)
    # rgb_range - default RGB range
    # alpha_range - default alpha range
def get_bounds(nr_poly, nr_vertex, size, rgb_range = [0,255], alpha_range = [0,100]):
    
    w_range = [1, size[0]]
    h_range = [1, size[1]]
    
    polygon = 3*[rgb_range] + [alpha_range] + nr_vertex*[w_range, h_range]
    bounds = nr_poly*polygon
    
    return bounds
    
    
# Generate new population randomly based on the bounds
    # bounds - bounds for each parameter of polygon
    # popsize - population size
    # nr_poly - number of polygons
    # nr_vertex - number of vertices
def popul_from_bounds(bounds, popsize, nr_poly, nr_vertex):
    
    population = []
    
    nr_params = 4 + 2*nr_vertex  # 4 - RGBA and then X,Y for vertices
    
    for i in range(0, popsize):
        indv = []

        for j in range(nr_poly):
            polygons = []
            for k in range(nr_params):
                polygons.append(randint(bounds[k][0],bounds[k][1]))
            indv.append(polygons)
        population.append(indv)
        
    return population
    
    
def show_img(size, dna, nr_poly):
    new_img = gen_image_poly(size, np.split(dna, nr_poly))
    plt.imshow(new_img)
    plt.show()

# Similar to show_img, but saves instead of showing
    # size - tuple - size of the image
    # dna - polygon parameters for generating image
    # nr_poly - how many points used
    # file - string - file name
def save_img(size, dna, nr_poly, file):
    new_img = gen_image_poly(size, np.split(dna, nr_poly))
    plt.imsave(file + ".png", new_img)
    
    
def hill_poly(img_name, nr_poly = 100, nr_vertex = 6, iter = 10000, mut_rate = 0.05, mut_chance = 0.05, \
                 save_every = 100, mut_random = False, path = "images/", file_name = "img"):
    
    #Load image
    img = Image.open(img_name).convert("RGB")
    orig_img = np.array(img)
    
    # Create dictinary
    if not os.path.exists(path):
        os.makedirs(path)
    
    file_name = path + file_name
    
    # Get params
    size = tuple(reversed(orig_img.shape[:2])) # W and H of image
    bounds = get_bounds(nr_poly, nr_vertex, size)
    # Init DNA
    dna = popul_from_bounds(bounds, popsize=1, nr_poly=nr_poly, nr_vertex=nr_vertex)[0]  # Only one element in pop, take 1st
    dna = np.array(dna).flatten()
    
    cost_last = cost_poly(nr_poly=nr_poly, orig_mat=orig_img, params=dna)
        
    len_dna = len(dna)
    dna_last = dna
    dna_new = dna_last.copy()
    
    impr = 0
    
    for i in range(iter):    
        
        # Select randomly param to change and change it by mutation rate
        idx = randint(0, len_dna - 1)
        direction = choice([-1,1])  # Select randomly the direction
        limits = bounds[idx]  # limits 
        
        if mut_random:
            dna_new[idx] = randint(limits[0], limits[1])
        else:
            #Mutate
            dna_new[idx] = dna_last[idx] + direction*limits[1]*mut_rate
        
        # Quick limits check
        if dna_new[idx] > limits[1]:
            dna_new[idx] = limits[1]
        elif dna_new[idx] < limits[0]:
            dna_new[idx] = limits[0]
        
        # Calculate new cost
        cost_new = cost_poly(nr_poly=nr_poly, orig_mat=orig_img, params=dna_new)
        
        mut = random()
        
        # Compare costs
        if (mut < mut_chance)  or (cost_new < cost_last):
            dna_last = dna_new.copy()
            cost_last = cost_new
            impr += 1
        else:
            dna_new = dna_last.copy()
            
        if i % save_every == 0:
            print("Iteration", i)
            print("improvements", impr)
            fit = cost_last/(255*255*3*size[0]*size[1])
            print("Fitness: ", 100-100*fit)
            save_img(size, dna_last, nr_poly, file_name + "_" + str(i))
        
        if i % (save_every*10) == 0:
            print("DNA:", ','.join(str(p) for p in dna_last) 

    
    
    save_img(size, dna_last, nr_poly, file_name + "_" + str(i))
    return dna_last