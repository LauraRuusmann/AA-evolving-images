from __future__ import print_function

from hill_voronoi import hill_voronoi
import argparse


# Call out main function
if __name__ == "__main__":
    print("Start evolving")
    
    # Parsing arguments for Hill climbing function
    ap = argparse.ArgumentParser()
    ap.add_argument('-img_name', default='images/darwin.png')
    ap.add_argument('-nr_points', type=int, default=30)
    ap.add_argument('-iter', type=int, default=10000)
    ap.add_argument('-mut_rate', type=float, default=0.05)
    ap.add_argument('-mut_chance', type=float, default=0.05)
    ap.add_argument('-save_every', type=int, default=100)
    ap.add_argument('-mut_random', type=bool, default=True)
    ap.add_argument('-path', default="images/")
    ap.add_argument('-file_name', default="img")
    ap.add_argument('-method', default="voronoi")


    args = vars(ap.parse_args())
    
    method = args.pop("method")
    
    ## Run lstm function
    if method == "voronoi":
        hill_voronoi(**args)
    else:
        print("Hill for polygons!")
    
    #python main_hill.py -img_name "images/darwin.png" -nr_points 30 -iter 10000 -mut_random 1 -mut_chance -1 -path "res/darwin/" -file_name="img"
    
    print("Finished")