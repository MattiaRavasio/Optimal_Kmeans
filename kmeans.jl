using CSV, Tables, LinearAlgebra, Random, Gurobi, JuMP, DataFrames, Statistics, MLJ, Plots, Clustering, Distances

"""
J is the number of points/record in the dataset
I is the number of clusters
D is the dimensionality of the points (number of features)
"""

function generate_points(I,J,D, std, seed = 1234)
    Random.seed!(seed);
    X, yy = make_blobs(J, D; centers=I, cluster_std=std)
    points = Matrix(DataFrame(X));
    min = minimum(points, dims=1);
    max = maximum(points, dims=1);
    points = (points .- min) ./ (max .- min);
    return points
end

function manhattan_optimal_kmenas(points, I, J, D)
    
    model = JuMP.Model(Gurobi.Optimizer);

    max_d = maximum(manhattan_distance(points))

    #set_optimizer_attribute(model, "TimeLimit", 300)

    @variable(model, gamma[1:J]);
    @variable(model, z[1:J, 1:I],Bin);
    @variable(model, mu[1:J, 1:I] >=0);
    @variable(model, r[1:J, 1:I] >=0);
    @variable(model, max_d >= x[1:I, 1:D] >=0);
    @variable(model, y[1:J, 1:D, 1:I] >=0);

    @constraint(model, [j = 1:J], sum(z[j,:]) == 1);

    @constraint(model, [i=1:I, j=1:J, d=1:D], sum(y[j,d,i] for d=1:D) <= r[j,i]);
    @constraint(model, [i=1:I, j=1:J, d=1:D], y[j,d,i] >= x[i,d] - points[j,d]);
    @constraint(model, [i=1:I, j=1:J, d=1:D], y[j,d,i] >= -(x[i,d] - points[j,d]));

    @constraint(model, [i=1:I, j=1:J], gamma[j]>= r[j,i]-mu[j,i]);

    @constraint(model, [i = 1:I, j = 1:J], max_d*(1-z[j,i]) >= mu[j,i]);

    @objective(model, Min, sum(gamma[j] for j=1:J));

    optimize!(model);

    return value.(x), [argmax(value.(z)[i,:]) for i = 1:J], value.(gamma)  # centers, assignments, obj_value
end

function euclidean_optimal_kmenas(points,  I, J, D)
    
    model = JuMP.Model(Gurobi.Optimizer);

    max_d = maximum(euclidean_distance(points));

    set_optimizer_attribute(model, "TimeLimit", 300)

    @variable(model, gamma[1:J]);
    @variable(model, z[1:J, 1:I], Bin);
    @variable(model, mu[1:J, 1:I] >=0);
    @variable(model, r[1:J, 1:I] >=0);
    @variable(model, max_d >= x[1:I, 1:D] >=0);
    @variable(model, y[1:J, 1:D, 1:I] >=0);

    @constraint(model, [j = 1:J], sum(z[j,:]) == 1);

    @constraint(model, [i=1:I, j=1:J, d=1:D], sum(y[j,d,i] for d=1:D) <= r[j,i]);
    @constraint(model,[i=1:I, j=1:J, d=1:D], [y[j,d,i]; x[i,d] - points[j,d]] in SecondOrderCone())

    @constraint(model, [i=1:I, j=1:J], gamma[j]>= r[j,i]-mu[j,i]);

    @constraint(model, [i = 1:I, j = 1:J], max_d*(1-z[j,i]) >= mu[j,i]);

    @objective(model, Min, sum(gamma[j] for j=1:J));

    optimize!(model);

    return value.(x), [argmax(value.(z)[i,:]) for i = 1:J], value.(gamma)  # centers, assignments, obj_value
end

function mean_silhouette_score(assignment, counts, points)
    distances = pairwise(Euclidean(), points')
    return mean(silhouettes(assignment, counts, distances))
end

function euclidean_distance(points)
    n,m = size(points)
    distances = ones((n,n))
    for i in 1:n
        for j in 1:n
            distances[i,j] = sqrt(sum((points[i,:] - points[j,:]).^2))
        end
    end
    return distances
end

function manhattan_distance(points)
    n,m = size(points)
    distances = ones((n,n))
    for i in 1:n
        for j in 1:n
            distances[i,j] = sum(abs.(points[i,:] - points[j,:]))
        end
    end
    return distances
end

function manhattan_warm_start(points, I, J, D)
    
    model = JuMP.Model(Gurobi.Optimizer);

    max_d = maximum(manhattan_distance(points))

    set_optimizer_attribute(model, "TimeLimit", 300)

    @variable(model, gamma[1:J]);
    @variable(model, z[1:J, 1:I],Bin);
    @variable(model, mu[1:J, 1:I] >=0);
    @variable(model, r[1:J, 1:I] >=0);
    @variable(model, max_d >= x[1:I, 1:D] >=0);
    @variable(model, y[1:J, 1:D, 1:I] >=0);

    km = kmeans(points', I);
    x_warm = km.centers';
    z_warm = zeros(J,I);
    assignments = km.assignments;
    for i = 1:J
        z_warm[i,assignments[i]] = 1
    end

    set_start_value.(z, z_warm);
    set_start_value.(x, x_warm);

    @constraint(model, [j = 1:J], sum(z[j,:]) == 1);

    @constraint(model, [i=1:I, j=1:J, d=1:D], sum(y[j,d,i] for d=1:D) <= r[j,i]);
    @constraint(model, [i=1:I, j=1:J, d=1:D], y[j,d,i] >= x[i,d] - points[j,d]);
    @constraint(model, [i=1:I, j=1:J, d=1:D], y[j,d,i] >= -(x[i,d] - points[j,d]));

    @constraint(model, [i=1:I, j=1:J], gamma[j]>= r[j,i]-mu[j,i]);

    @constraint(model, [i = 1:I, j = 1:J], max_d*(1-z[j,i]) >= mu[j,i]);

    @objective(model, Min, sum(gamma[j] for j=1:J));

    optimize!(model);

    return value.(x), [argmax(value.(z)[i,:]) for i = 1:J], value.(gamma)  # centers, assignments, obj_value
end


function euclidean_warm_start(points,  I, J, D)
    
    model = JuMP.Model(Gurobi.Optimizer);

    max_d = maximum(euclidean_distance(points));

    set_optimizer_attribute(model, "TimeLimit", 300)

    @variable(model, gamma[1:J]);
    @variable(model, z[1:J, 1:I], Bin);
    @variable(model, mu[1:J, 1:I] >=0);
    @variable(model, r[1:J, 1:I] >=0);
    @variable(model, max_d >= x[1:I, 1:D] >=0);
    @variable(model, y[1:J, 1:D, 1:I] >=0);

    km = kmeans(points', I);
    x_warm = km.centers';
    z_warm = zeros(J,I);
    assignments = km.assignments;
    for i = 1:J
        z_warm[i,assignments[i]] = 1
    end

    @constraint(model, [j = 1:J], sum(z[j,:]) == 1);

    @constraint(model, [i=1:I, j=1:J, d=1:D], sum(y[j,d,i] for d=1:D) <= r[j,i]);
    @constraint(model,[i=1:I, j=1:J, d=1:D], [y[j,d,i]; x[i,d] - points[j,d]] in SecondOrderCone())

    @constraint(model, [i=1:I, j=1:J], gamma[j]>= r[j,i]-mu[j,i]);

    @constraint(model, [i = 1:I, j = 1:J], max_d*(1-z[j,i]) >= mu[j,i]);

    @objective(model, Min, sum(gamma[j] for j=1:J));

    optimize!(model);

    return value.(x), [argmax(value.(z)[i,:]) for i = 1:J], value.(gamma)  # centers, assignments, obj_value
end


# """
# J is the number of points/record in the dataset
# I is the number of clusters
# D is the dimensionality of the points (number of features)
# # """

# dim3_optimal_performance = []
# dim3_warm_start_performance = []

# for J in [20,50,100,300,500]
#     for I in [2,3,4,5]
#         for D = 3
#             points = generate_points(I,J,D, ones(I));
#             centroids, assignment, obj_value = euclidean_optimal_kmenas(points, I,J,D);
#             push!(dim3_optimal_performance, mean_silhouette_score(assignment, counts(assignment), points))
#             centroids, assignment, obj_value = euclidean_warm_start(points, I,J,D);
#             push!(dim3_warm_start_performance, mean_silhouette_score(assignment, counts(assignment), points))
#         end
#     end
# end

# dim3_optimal_performance = rename!(DataFrame(reshape(dim3_optimal_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
# dim3_optimal_performance[!,:centers] = ["2","3","4","5"];

# dim3_warm_start_performance = rename!(DataFrame(reshape(dim3_warm_start_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
# dim3_warm_start_performance[!,:centers] = ["2","3","4","5"];

# CSV.write("results/dim3_optimal_euclidean_performance.csv",dim3_optimal_performance)
# CSV.write("results/dim3_warm_start_euclidean_performance.csv",dim3_warm_start_performance)

# dim3_optimal_performance = []
# dim3_warm_start_performance = []

# for J in [20,50,100,300,500]
#     for I in [2,3,4,5]
#         for D = 3
#             points = generate_points(I,J,D, ones(I));
#             centroids, assignment, obj_value = manhattan_optimal_kmenas(points, I,J,D);
#             push!(dim3_optimal_performance, mean_silhouette_score(assignment, counts(assignment), points))
#             centroids, assignment, obj_value = manhattan_warm_start(points, I,J,D);
#             push!(dim3_warm_start_performance, mean_silhouette_score(assignment, counts(assignment), points))
#         end
#     end
# end

# dim3_optimal_performance = rename!(DataFrame(reshape(dim3_optimal_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
# dim3_optimal_performance[!,:centers] = ["2","3","4","5"];

# dim3_warm_start_performance = rename!(DataFrame(reshape(dim3_warm_start_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
# dim3_warm_start_performance[!,:centers] = ["2","3","4","5"];

# CSV.write("results/dim3_optimal_manhattan_performance.csv",dim3_optimal_performance)
# CSV.write("results/dim3_warm_start_manhattan_performance.csv",dim3_warm_start_performance)

# dim3_optimal_performance = []

# for J in [20,50,100,300,500]
#     for I in [2,3,4,5]
#         for D = 3
#             points = generate_points(I,J,D, ones(I));
#             km = kmeans(points', I)
#             assignments = km.assignments;
#             push!(dim3_optimal_performance, mean_silhouette_score(assignments, counts(assignments), points))
#         end
#     end
# end

# dim3_optimal_performance = rename!(DataFrame(reshape(dim3_optimal_performance, (4,5)), :auto),:x1 => :"20", :x2 => :"50", :x3 => :"100", :x4 => :"300", :x5 => :"500")
# dim3_optimal_performance[!,:centers] = ["2","3","4","5"];

# CSV.write("results/dim3_kmeans_performance.csv",dim3_optimal_performance)

# points = generate_points(3,100,2, ones(3));

# @time centroids, assignment, obj_value = manhattan_optimal_kmenas(points, 3,100,2);
# shil = mean_silhouette_score(assignment, counts(assignment), points);
# println("The mean silhouette score for the optimal manhattan kmeans is $shil")

# km = kmeans(points', 3)
# assignments = km.assignments;
# shil = mean_silhouette_score(assignments, counts(assignments), points);
# println("The mean silhouette score for the kmeans is $shil")