from random import randint, choice, random, sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math
from datetime import datetime



# Calculate the bounds for voronoi DNA
def get_bounds_voronoi(nr_points, size, rgb_range=[0, 255]):
    w_range = [1, size[0]]
    h_range = [1, size[1]]

    point = 3 * [rgb_range] + [w_range, h_range]  # R; G, B, x, y
    bounds = nr_points * point

    return bounds


# Generate Population based on bounds for voronoi
def popul_from_bounds_voronoi(bounds, popsize, nr_params=5):
    population = []

    nr_points = len(bounds) // nr_params  # 5 - RGB and then X,Y

    for i in range(0, popsize):
        indv = []

        for j in range(nr_points):
            point = []
            for k in range(nr_params):
                point.append(randint(bounds[k][0], bounds[k][1]))
            indv.append(point)
        population.append(indv)

    return population


#todo this script has key index error
## NOTE this function copied from - stackoverflow - https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        if p1 in all_ridges:  # Check if key in dictionary (previously got error)
            ridges = all_ridges[p1]
        else:
            continue

        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def gen_image_voronoi(size, vor_data):
    points = [x[3:] for x in vor_data]  # X,Y
    colors = [x[:3] for x in vor_data]  # [(R,G,B),..]

    # Calculate voronoi diagram based on points
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Split into X and Y
    X = [coord[0] for coord in vertices]
    Y = [coord[1] for coord in vertices]
    # Get padding for X and Y
    pad_x = abs(math.floor(min(X)))
    pad_y = abs(math.floor(min(Y)))
    # Normalize the data, so it starts from 0
    X = np.array([X - min(X)])
    Y = np.array([Y - min(Y)])
    # Back to vertices
    vertices = np.concatenate((X.T, Y.T), axis=1)
    size_new = (math.ceil(max(X[0])), math.ceil(max(Y[0])))

    image = Image.new("RGB", size_new, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image, mode="RGBA")


    for i, region in enumerate(regions):
        new_color = tuple(colors[i])
        polygon = vertices[region]
        polygon = tuple(map(tuple, polygon))
        draw.polygon(polygon, fill=new_color)

    img_mat = np.array(image)

    del image
    del draw

    return img_mat[pad_y:size[1] + pad_y, pad_x:size[0] + pad_x]


# Calculates square error between two images
# img1, img2 - images in array with shape (w,h,3)
def square_error(img1, img2):
    return np.sum(np.square(img1.astype("int64") - img2.astype("int64")))


# Cost function for using polygons
# orig_mat - original image given as matrix
# nr_points - number of points used
# params - information about voronoi data in 1-D array (R,G,B,x,y)
def cost_voronoi(params, orig_mat, nr_points):
    vor_data = np.split(params, nr_points)
    size = tuple(reversed(orig_mat.shape[:2]))  # W and H of image

    new_mat = gen_image_voronoi(size, vor_data)

    return square_error(orig_mat, new_mat)


# Show image generated base on dna
# size - tuple - size of the image
# dna - voronoi parameters for generating image
# nr - how many points used
def show_img(size, dna, nr):
    new_img = gen_image_voronoi(size, np.split(dna, nr))
    plt.imshow(new_img)
    plt.show()


# Similar to show_img, but saves instead of showing
# size - tuple - size of the image
# dna - voronoi parameters for generating image
# nr - how many points used
# file - string - file name
def save_img(size, dna, nr, file):
    new_img = gen_image_voronoi(size, np.split(dna, nr))
    plt.imsave(file + ".png", new_img)

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

def differential_evolution(img_name, nr_points,popsize=50, iter=4000, mutate=0.5, recombination=0.7):

    # Load image
    img = Image.open(img_name).convert("RGB")
    orig_mat = np.array(img)


    size = tuple(reversed(orig_mat.shape[:2]))  # W and H of image


    bounds = get_bounds_voronoi(nr_points=nr_points, size=size)


    population = []
    for i in range(0, popsize):
        indv = []
        for j in range(nr_points):
            polygons = []
            for k in range(5):
                polygons.append(randint(bounds[k][0], bounds[k][1]))
            indv.append(polygons)
        population.append(indv)

    for i in range(1, iter + 1):
        gen_scores = []  # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates = list(range(0,popsize)) #[*range(0, popsize)]
            canidates.remove(j)
            random_index = sample(canidates, 3)

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
                donor = [int(x_1_i + mutate * x_diff_i) for x_1_i, x_diff_i in zip(x_1[v], x_diff[v])]
                v_donor.append(donor)


            v_donor = ensure_bounds(v_donor, bounds)

            v_trial = []
            # cycle through each variable in our target vector
            for k in range(len(x_t)):
                crossover = random()
                # recombination occurs when crossover <= recombination rate
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                # recombination did not occur
                else:
                    v_trial.append(x_t[k])
            #print(j)
            score_trial = cost_voronoi(np.array(v_trial).flatten(), orig_mat, nr_points)
            score_target = cost_voronoi(np.array(x_t).flatten(), orig_mat, nr_points)

            if score_trial < score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)

            else:
                gen_scores.append(score_target)
        # gen_avg = sum(gen_scores) / popsize  # current generation avg. fitness
        # gen_best = min(gen_scores)  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]  # solution of best individual
        #new_img = gen_image_voronoi(size, gen_sol)
        print(i)
        #plt.imshow(new_img)
        #plt.show()


    date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    outputfile = "voronoi-" + str(date) + "-" + str(img_name.split("/")[2].split(".")[0]) + "-iter" + str(iter) + "-points" + str(nr_points) + ".png"
    outputimage = gen_image_voronoi(size, gen_sol)
    plt.imsave(outputfile, outputimage)
    #return gen_sol


differential_evolution("AA-evolving-images/images/darwin.PNG",75)


