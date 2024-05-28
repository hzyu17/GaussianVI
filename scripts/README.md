## A simple 1-dimensional nonlinear estimation example [1]
A simple 1D example nonlinear posterior is under src folder. 

## Problem
Assume that the true state of a robot is drawn from a Gaussian distribution
$$x \sim \mathcal{N}(\mu_p, \sigma_p^2).$$

Generate an artificial measurement following
$$y = \frac{fb}{x}+n, ~ ~ n\sim \mathcal{N}(0, \sigma_r^2).$$

The two factors (1 Gaussian and 1 nonlinear) are thus defined as 
$$\phi = \frac{1}{2}\frac{(x-\mu_p)^2}{\sigma_p^2}, ~ ~ \psi = \frac{1}{2} \frac{(y-\frac{fb}{x})^2}{\sigma_r^2}.$$

The cost function in the variational inference problem is 
$$V(q) = \mathbb{E}_q [\phi + \psi] + \frac{1}{2}\log(\sigma^{-2})$$
for a proposal Gaussian distribution $$q\sim\mathcal{N}(\mu, \sigma^2).$$

**To run the code for this example**
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

<a href="figures/1d_NGD_gt.pdf" target="_blank">View the Extected output figure</a>


## References
<a id="1">[1]</a> 
Barfoot, Timothy D. State estimation for robotics. Cambridge University Press, 2024.

