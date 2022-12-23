# Optimal_Kmeans

## Project description

This is a repository for the Machine learning under modern optimization lenses  group project. We attempted solving the stability and provable optimality issues that affect the most used clustering algorithm in the world, K-means. In order to do this we got rid of the random nature of the algorithm and leverage the used of Mixed-Ineteger programming and Gurobi/JuMP.  

The code is mainly wirtten in Julia, with some peripheral experimentation in python. 

## The repository

- **Documents**: presentation and report for the project
- **Results**: some results comparing the warm-start and cold-start algoritmhs to the classical kmeans framework, in low dimensionality spaces
- **Experimentation.ipynb**: some experimentions and plottings 
- **Autoencoder.ipynb**: experimentations with dimensionality reduction
- **Kmeans.jl**: formulations proposed 

## License

[MIT](https://choosealicense.com/licenses/mit/)
