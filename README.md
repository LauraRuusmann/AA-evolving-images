# AA-evolving-images
A project for advanced algorithms class in the University of Tartu.

## Introduction

Optimization algorithms have been proven to be useful for many applications. In given project differential evolution and hill-climbing algorithms were used to evolve images from random polygons and Voronoi points. Meaning that with each iteration the algorithm tries to get closer to the original image, however its possibilities are limited with the number of polygons or with the number of Voronoi points.

## Metaheuristic algorithms used

We have used hill climibing and differential evolution algorithms for finding the best solutions. 
  
Hill climbing is a local search algorithm. In this project one parameter was changed to random value at a time. After changing the value, the newly generated image was compared to original. If it was better than previous, it was kept, otherwise discarded.
  
In evolutionary computation, differential evolution is a method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. In our implementation, in each iteration each candidate solution in the population is mutated and the mutated population entity is accepted according to the recombination rate.

## Voronoi points and polygons

Original image:

![Original image](https://github.com/LauraRuusmann/AA-evolving-images/blob/master/images/eiffel.jpg)

Polygons: each polygon had separate color, alpha and coordinate values.

![Example of evolving with polygons](https://github.com/LauraRuusmann/AA-evolving-images/blob/master/results_poly/e300_99999.png)

Voronoi diagram: these diagrams were built using points that had color values.

![Example of evolving Voronoi cells](https://github.com/LauraRuusmann/AA-evolving-images/blob/master/results_vor/eiffel400_24900.png)

## Timelapse

Examples of the results' progress can be seen [here](https://imgur.com/a/4lHuE) (external link to Imgur).

## Poster

We have created a descriptive poster for our project which can be found [here](https://github.com/LauraRuusmann/AA-evolving-images/blob/master/poster.pdf).
