## A simple 1-dimensional nonlinear estimation example
A simple 1D example nonlinear posterior is under src folder. 

**To run the code**
```
mkdir build && cd build
cmake ..
make
./src/1d_example
```
This will run the experiment and save all the optimization results under the data/1d folder.

**To visualize** the groundtruth cost map, the iteration results, and the total cost of each iteration, run the jupyter notebook 
```
scripts/plot1dexample.ipynb
``` 

<a href="path/to/document.pdf" target="_blank">View the Extected output figure</a>