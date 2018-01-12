from __future__ import print_function

from scripts.hill_voronoi import hill_voronoi
from scripts.hill_poly import hill_poly

import argparse
import time


# Call out main function
if __name__ == "__main__":
    print("Start evolving")
    
    # Parsing arguments for Hill climbing function
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_name', default='images/darwin.png', help="Path to original image")
    ap.add_argument('-nr_points', type=int, default=200, help="Number of Voronoi points (method=voronoi)")
    ap.add_argument('-nr_poly', type=int, default=200, help="Number of polygons (method=polygon)")
    ap.add_argument('-nr_vertex', type=int, default=6, help="Number of vertices (method=polygon)")

    ap.add_argument('-iter', type=int, default=30000, help="Number of iterations")
    ap.add_argument('-mut_rate', type=float, default=0.05, help="Mutation speed, does not change over time.")
    ap.add_argument('-mut_chance', type=float, default=0.05, help="Change of accepting worsening results.")
    ap.add_argument('-save_every', type=int, default=100, help="Save result image after every i-th iteration.")
    ap.add_argument('-mut_random', type=bool, default=True, help="Random mutations, if it's TRUE, then variable 'mut_rate' ignored.")
    ap.add_argument('-path', default="res_image/", help="Path to folder, where results are saved")
    ap.add_argument('-file_name', default="img", help="Name of result image files (currently only png format used)")
    ap.add_argument('-method', default="polygon", help="Methods (styles) used: [voronoi,polygon]")
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