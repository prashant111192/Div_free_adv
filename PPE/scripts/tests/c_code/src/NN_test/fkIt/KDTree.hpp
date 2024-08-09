#include <iostream>
#include <vector>
#include <algorithm>


// Define a structure for the points
// struct Point {
//     std::vector<double> coordinates;
//     unsigned int idx;
//     Point(std::initializer_list<double> init, size_t idx) : coordinates(init), idx(unsigned(idx)){}
// };

// Define a structure for the nodes of the k-d tree
struct KDNode {
    std::vector<std::vector <double>> point;
    KDNode* left;
    KDNode* right;
    unsigned int idx;

    KDNode(std::vector<double> pt, size_t idx) : left(nullptr), right(nullptr), idx((unsigned int)idx){
        point.push_back(pt);
    }
};

// Function to build the k-d tree
KDNode* buildKDTree(std::vector<std::vector<double>>& points, int depth = 0) {
    if (points.empty()) return nullptr;

    // Select axis based on depth so that axis cycles through all valid dimensions
    size_t k = points[0].size();
    int axis = depth % k;

    //     // Sort point list and choose median as pivot element
    // std::sort(points.begin(), points.end(), [axis](const std::pair<std::vector<double>, unsigned int>& a, const std::pair<std::vector<double>, unsigned int>& b) {
    //     return a.first[axis] < b.first[axis];
    // });
    // Sort point list and choose median as pivot element
    std::sort(points.begin(), points.end(), [axis](const std::vector<double> a, const std::vector<double> b) {
        return a[axis] < b[axis];
    });

    // Find the median
    size_t medianIndex = points.size() / 2;
    std::vector<double> medianPoint = points[medianIndex];

    // Create node and construct subtrees
    KDNode* node = new KDNode(medianPoint, medianIndex);
    std::vector<std::vector<double>> leftPoints(points.begin(), points.begin() + medianIndex);
    std::vector<std::vector<double>> rightPoints(points.begin() + medianIndex + 1, points.end());

    node->left = buildKDTree(leftPoints, depth + 1);
    node->right = buildKDTree(rightPoints, depth + 1);

    return node;
}

double squaredDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

// Function to search the k-d tree and collect neighbors within the specified radius
void findNeighbors(KDNode* node,
                    const std::vector<double>& target, 
                    double radius, 
                    int depth, 
                    std::vector<std::vector<double>>& neighbors, 
                    std::vector<double>& distances, 
                    std::vector<unsigned>& nn_idx) 
{
    if (node == nullptr) return;

    // Calculate the squared radius to avoid computing square roots
    double radiusSquared = radius * radius;

    // Check if the current node's point is within the radius
    if (squaredDistance(node->point[0], target) <= radiusSquared) {
        neighbors.push_back(node->point[0]);
        distances.push_back(squaredDistance(node->point[0], target));
        nn_idx.push_back(node->idx);
    }

    // Select axis based on depth
    size_t k = target.size();
    int axis = depth % k;

    // Determine whether to search the left or right subtree or both
    double diff = target[axis] - node->point[0][axis];
    double diffSquared = diff * diff;

    if (diff <= 0) {
        findNeighbors(node->left, target, radius, depth + 1, neighbors, distances, nn_idx);
        if (diffSquared <= radiusSquared) {
            findNeighbors(node->right, target, radius, depth + 1, neighbors, distances, nn_idx);
        }
    } else {
        findNeighbors(node->right, target, radius, depth + 1, neighbors, distances, nn_idx);
        if (diffSquared <= radiusSquared) {
            findNeighbors(node->left, target, radius, depth + 1, neighbors, distances, nn_idx);
        }
    }
}
// Function to print the k-d tree (for debugging purposes)
void printKDTree(KDNode* node, int depth = 0) {
    if (node == nullptr) return;

    std::cout << std::string(depth, '\t') << "(";
    for (size_t i = 0; i < node->point[0].size(); ++i) {
        std::cout << node->point[0][i];
        if (i < node->point[0].size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;

    printKDTree(node->left, depth + 1);
    printKDTree(node->right, depth + 1);
}

