from __future__ import print_function

from hill_voronoi import hill_voronoi
from hill_poly import hill_poly

import argparse
import time


# Call out main function
if __name__ == "__main__":
    print("Start evolving")
    
    # Parsing arguments for Hill climbing function
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_name', default='images/darwin.png')
    ap.add_argument('-nr_points', type=int, default=100)
    ap.add_argument('-nr_poly', type=int, default=100)
    ap.add_argument('-nr_vertex', type=int, default=6)

    ap.add_argument('-iter', type=int, default=10000)
    ap.add_argument('-mut_rate', type=float, default=0.05)
    ap.add_argument('-mut_chance', type=float, default=0.05)
    ap.add_argument('-save_every', type=int, default=100)
    ap.add_argument('-mut_random', type=bool, default=True)
    ap.add_argument('-path', default="images/")
    ap.add_argument('-file_name', default="img")
    ap.add_argument('-method', default="voronoi")
    args = vars(ap.parse_args())
    
    print("Method:", args["method"])
    print("Image:", args["img_name"])
    print("Points:", args["nr_points"])
    print("Polygons:", args["nr_poly"])
    print("Vertices:", args["nr_vertex"])
    
    method = args.pop("method")
    
    start = time.time()  # Time script
    
    ## Run lstm function
    if method == "voronoi":
        args.pop("nr_poly")
        args.pop("nr_vertex")
        hill_voronoi(**args)
    else:
        args.pop("nr_points")
        hill_poly(**args)
    
    #python main_hill.py -img_name "images/darwin.png" -nr_points 30 -iter 10000 -mut_random 1 -mut_chance -1 -path "res/darwin/" -file_name="img"
    
    end = time.time()
    elapsed = end - start

    print("Time taken: %0.3f minutes" % (elapsed/60))