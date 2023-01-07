using CSV, Tables, LinearAlgebra, Random, Gurobi, JuMP, DataFrames, Statistics, MLJ, Plots, Clustering, Distances

"""
J is the number of points/record in the dataset
I is the number of clusters
D is the dimensionality of the points (number of features)
"""

include("kmeans.jl")

dim3_optimal_performance = []
dim3_warm_start_performance = []

for J in [20,50,100,300,500]
    for I in [2,3,4,5]
        for D = 3
            points = generate_points(I,J,D, ones(I));
            centroids, assignment, obj_value = euclidean_optimal_kmenas(points, I,J,D);
            push!(dim3_optimal_performance, mean_silhouette_score(assignment, counts(assignment), points))
            centroids, assignment, obj_value = euclidean_warm_start(points, I,J,D);
            push!(dim3_warm_start_performance, mean_silhouette_score(assignment, counts(assignment), points))
        end
    end
end

dim3_optimal_performance = rename!(DataFrame(reshape(dim3_optimal_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
dim3_optimal_performance[!,:centers] = ["2","3","4","5"];

dim3_warm_start_performance = rename!(DataFrame(reshape(dim3_warm_start_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
dim3_warm_start_performance[!,:centers] = ["2","3","4","5"];

CSV.write("results/dim3_optimal_euclidean_performance.csv",dim3_optimal_performance)
CSV.write("results/dim3_warm_start_euclidean_performance.csv",dim3_warm_start_performance)

dim3_optimal_performance = []
dim3_warm_start_performance = []

for J in [20,50,100,300,500]
    for I in [2,3,4,5]
        for D = 3
            points = generate_points(I,J,D, ones(I));
            centroids, assignment, obj_value = manhattan_optimal_kmenas(points, I,J,D);
            push!(dim3_optimal_performance, mean_silhouette_score(assignment, counts(assignment), points))
            centroids, assignment, obj_value = manhattan_warm_start(points, I,J,D);
            push!(dim3_warm_start_performance, mean_silhouette_score(assignment, counts(assignment), points))
        end
    end
end

dim3_optimal_performance = rename!(DataFrame(reshape(dim3_optimal_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
dim3_optimal_performance[!,:centers] = ["2","3","4","5"];

dim3_warm_start_performance = rename!(DataFrame(reshape(dim3_warm_start_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
dim3_warm_start_performance[!,:centers] = ["2","3","4","5"];

CSV.write("results/dim3_optimal_manhattan_performance.csv",dim3_optimal_performance)
CSV.write("results/dim3_warm_start_manhattan_performance.csv",dim3_warm_start_performance)

dim3_optimal_performance = []

for J in [20,50,100,300,500]
    for I in [2,3,4,5]
        for D = 3
            points = generate_points(I,J,D, ones(I));
            km = kmeans(points', I)
            assignments = km.assignments;
            push!(dim3_optimal_performance, mean_silhouette_score(assignments, counts(assignments), points))
        end
    end
end

dim3_optimal_performance = rename!(DataFrame(reshape(dim3_optimal_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
dim3_optimal_performance[!,:centers] = ["2","3","4","5"];

CSV.write("results/dim3_kmeans_performance.csv",dim3_optimal_performance)

points = generate_points(3,100,2, ones(3));

@time centroids, assignment, obj_value = manhattan_optimal_kmenas(points, 3,100,2);
shil = mean_silhouette_score(assignment, counts(assignment), points);
println("The mean silhouette score for the optimal manhattan kmeans is $shil")

km = kmeans(points', 3)
assignments = km.assignments;
shil = mean_silhouette_score(assignments, counts(assignments), points);
println("The mean silhouette score for the kmeans is $shil")