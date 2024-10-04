/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2024 cfdARCHO

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <numeric>
#include <random>
#include <iostream>
#include <matplot/freestanding/plot.h>
#include "distribution_algo.hpp"
#include "fvm.hpp"

std::vector<int> linear_distribution(Mesh2D* mesh, size_t num_proc, std::vector<size_t>& priorities) {
    size_t num_modes_per_proc = std::ceil(static_cast<double>(mesh->_num_nodes) / num_proc);
    int current_proc = -1;
    size_t current_proc_num = 0;

    std::vector<int> node_distribution{};

    for (size_t i = 0; i < mesh->_num_nodes; ++i) {
        if (i % num_modes_per_proc == 0) {
            current_proc += 1;
            node_distribution.emplace_back();
            current_proc_num = 0;
        }
        node_distribution.push_back(current_proc);
        current_proc_num += 1;
    }
    return node_distribution;
}

void show_distribution(Mesh2D* mesh, std::vector<int>& distribution) {
    MatrixX4dRB grid_hist = {mesh->_x, mesh->_y};
    for (int i = 0; i < mesh->_num_nodes; ++i) {
        double value = distribution.at(i);
        size_t x_coord = mesh->_nodes.at(i)->_id / mesh->_x;
        size_t y_coord = mesh->_nodes.at(i)->_id % mesh->_x;
        grid_hist(x_coord, y_coord) = value;
    }
//    for (int i = 0; i < distribution.size(); ++i) {
//        std::cout << distribution.at(i) << " ";
//    }
//    std::cout << std::endl;
    auto vect = from_eigen_matrix<double>(grid_hist);
    matplot::image(vect, true);
    matplot::colorbar();
    matplot::show();
}

std::vector<int> cluster_distribution(Mesh2D* mesh, size_t num_proc, std::vector<size_t>& priorities) {
    std::vector<int> node_distribution(mesh->_num_nodes);

    auto sum_of_elems = std::accumulate(priorities.begin(), priorities.end(), size_t{0});
    size_t one_part_cluster_size = std::floor(static_cast<double>(mesh->_num_nodes) / static_cast<double>(sum_of_elems));
    std::cout << "one_part_cluster_size = " << one_part_cluster_size << std::endl;
    std::vector<size_t> cluster_sizes;
    for (auto priority : priorities) {
        cluster_sizes.push_back(priority * one_part_cluster_size);
    }

    auto num_of_elems = mesh->_num_nodes;

    Eigen::VectorXi point_to_cluster{num_of_elems};
    point_to_cluster.setConstant(0);
    Eigen::MatrixXd centroids{num_proc, 2};
    centroids.setConstant(0);
    Eigen::MatrixXd last_centroids{num_proc, 2};
    last_centroids.setConstant(0);
    Eigen::MatrixXd points{num_of_elems, 2};
    points.setConstant(0);
    Eigen::MatrixXd distances_to_centroids{num_of_elems, num_proc};
    distances_to_centroids.setConstant(0);
    Eigen::MatrixXd update_errors{num_proc, 2};
    update_errors.setConstant(0);

    for (int i = 0; i < num_of_elems; ++i) {
        points(i, 0) = mesh->_nodes.at(i)->x();
        points(i, 1) = mesh->_nodes.at(i)->y();
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, num_of_elems);

    std::vector<size_t> initial_clusters_ids{};
//    while (initial_clusters_ids.size() < num_proc) {
//        auto curr_id = distr(gen);
//        if (!std::count(initial_clusters_ids.begin(), initial_clusters_ids.end(), curr_id)) {
//            initial_clusters_ids.push_back(curr_id);
//        }
//    }
    int base_step = std::floor(static_cast<double>(num_of_elems) / static_cast<double>(num_proc));
    for (int i = 0; i < num_proc; ++i) {
        initial_clusters_ids.push_back(i * base_step);
    }

    for (int i = 0; i < num_proc; ++i) {
        centroids(i, 0) = mesh->_nodes.at(initial_clusters_ids[i])->x();
        centroids(i, 1) = mesh->_nodes.at(initial_clusters_ids[i])->y();
    }

    double last_error = 10;
    while (last_error > 0.05) {
        std::cout << "last_error = " << last_error << std::endl;
        last_centroids = centroids;

//        calculating distances
        for (int i = 0; i < num_of_elems; ++i) {
            for (int j = 0; j < num_proc; ++j) {
                double dist = std::sqrt(std::pow(points(i, 0) - centroids(j, 0), 2) + std::pow(points(i, 1) - centroids(j, 1), 2));
                distances_to_centroids(i, j) = dist;
            }
        }

//        calculating cluster assignments
        std::vector<int> current_cluster_sizes (num_proc, 0);
        for (int i = 0; i < num_of_elems; ++i) {
            auto point_distances = distances_to_centroids.row(i);
            std::vector<std::pair<size_t, double>> vector_pair_dist_cluster;
            for (int j = 0; j < num_proc; ++j) {
                double dist = point_distances(j);
                vector_pair_dist_cluster.emplace_back(j, dist);
            }

            std::sort(vector_pair_dist_cluster.begin(), vector_pair_dist_cluster.end(),
                      [](std::pair<size_t, double>& lhs, std::pair<size_t, double>& rhs){return rhs.second > lhs.second;});

            bool flag = false;
            for (int j = 0; j < num_proc; ++j) {
                size_t cluster_id = vector_pair_dist_cluster.at(j).first;
                if (current_cluster_sizes[cluster_id] < cluster_sizes[cluster_id]) {
                    current_cluster_sizes[cluster_id] = current_cluster_sizes[cluster_id] + 1;
                    point_to_cluster(i) = cluster_id;
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                point_to_cluster(i) = -1;
            }
        }

//        updating centroids
        for (int i = 0; i < num_proc; ++i) {
            centroids(i, 0) = 0;
            centroids(i, 1) = 0;
        }
        for (int i = 0; i < num_of_elems; ++i) {
            auto cluster_id = point_to_cluster(i);
            if (cluster_id >= 0) {
                centroids(cluster_id, 0) = centroids(cluster_id, 0) + points(i, 0);
                centroids(cluster_id, 1) = centroids(cluster_id, 1) + points(i, 1);
            }
        }
        for (int i = 0; i < num_proc; ++i) {
            centroids(i, 0) = centroids(i, 0) / current_cluster_sizes[i];
            centroids(i, 1) = centroids(i, 1) / current_cluster_sizes[i];
        }

//        calculate error
        for (int i = 0; i < num_proc; ++i) {
            update_errors(i, 0) = last_centroids(i, 0) - centroids(i, 0);
            update_errors(i, 1) = last_centroids(i, 1) - centroids(i, 1);
        }
        update_errors = update_errors.cwiseProduct(update_errors);
        auto update_errors_sum = update_errors.sum();
        last_error = std::sqrt(update_errors_sum / static_cast<double>(num_proc));
    }

//    finally assign points
    for (int i = 0; i < num_of_elems; ++i) {
        for (int j = 0; j < num_proc; ++j) {
            distances_to_centroids(i, j) = std::sqrt(std::pow(points(i, 0) - centroids(j, 0), 2) + std::pow(points(i, 1) - centroids(j, 1), 2));
        }
    }

    for (int i = 0; i < num_of_elems; ++i) {
        auto point_distances = distances_to_centroids.row(i);
        std::vector<std::pair<size_t, double>> vector_pair_dist_cluster{};
        for (int j = 0; j < num_proc; ++j) {
            double dist = point_distances(j);
            vector_pair_dist_cluster.emplace_back(j, dist);
        }

        std::sort(vector_pair_dist_cluster.begin(), vector_pair_dist_cluster.end(),
                  [](std::pair<size_t, double>& lhs, std::pair<size_t, double>& rhs){return rhs.second > lhs.second;});

        point_to_cluster(i) = vector_pair_dist_cluster.at(0).first;
    }

    for (int i = 0; i < num_of_elems; ++i) {
        auto cluster_id = point_to_cluster(i);
        node_distribution[i] = cluster_id;
    }

    std::cout << "Node clusters centroids: ";
    for (int i = 0; i < num_proc; ++i) {
        std::cout << centroids(i, 0) << ";" << centroids(i, 0) << "  ";
    }


//    show_distribution(mesh, node_distribution);

    return node_distribution;
}