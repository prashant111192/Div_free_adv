
/*
 * file: KDTree.hpp
 * author: J. Frederico Carvalho
 *
 * This is an adaptation of the KD-tree implementation in rosetta code
 *  https://rosettacode.org/wiki/K-d_tree
 * It is a reimplementation of the C code using C++.
 * It also includes a few more queries than the original
 *
 */


/*
* Chnages made to the original code:
* 1. Made changes to some function to allow returning the distance and the index.
* 2. Changed the index from size_t to unsigned int to reduce memory usage. The max value of unsigned int is 4,294,967,295
*/

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <iostream>

using point_t = std::vector< double >;
using indexArr = std::vector< unsigned int >;
using pointIndex = typename std::pair< std::vector< double >, unsigned int >;

class KDNode {
   public:
    using KDNodePtr = std::shared_ptr< KDNode >;
    unsigned int index;
    point_t x;
    KDNodePtr left;
    KDNodePtr right;

    // initializer
    KDNode();
    KDNode(const point_t &, const unsigned int &, const KDNodePtr &,
           const KDNodePtr &);
    KDNode(const pointIndex &, const KDNodePtr &, const KDNodePtr &);
    ~KDNode();

    // getter
    double coord(const unsigned int &);

    // conversions
    explicit operator bool();
    explicit operator point_t();
    explicit operator unsigned int();
    explicit operator pointIndex();
};

using KDNodePtr = std::shared_ptr< KDNode >;

KDNodePtr NewKDNodePtr();

// square euclidean distance
inline double dist2(const point_t &, const point_t &);
inline double dist2(const KDNodePtr &, const KDNodePtr &);

// euclidean distance
inline double dist(const point_t &, const point_t &);
inline double dist(const KDNodePtr &, const KDNodePtr &);

// Need for sorting
class comparer {
   public:
    unsigned int idx;
    explicit comparer(unsigned int idx_);
    inline bool compare_idx(
        const std::pair< std::vector< double >, unsigned int > &,  //
        const std::pair< std::vector< double >, unsigned int > &   //
    );
};

using pointIndexArr = typename std::vector< pointIndex >;

inline void sort_on_idx(const pointIndexArr::iterator &,  //
                        const pointIndexArr::iterator &,  //
                        unsigned int idx);

using pointVec = std::vector< point_t >;

class KDTree {
    KDNodePtr root;
    KDNodePtr leaf;

    KDNodePtr make_tree(const pointIndexArr::iterator &begin,  //
                        const pointIndexArr::iterator &end,    //
                        const unsigned int &length,                  //
                        const unsigned int &level                    //
    );

   public:
    KDTree() = default;
    explicit KDTree(pointVec point_array);

   private:
    KDNodePtr nearest_(           //
        const KDNodePtr &branch,  //
        const point_t &pt,        //
        const unsigned int &level,      //
        const KDNodePtr &best,    //
        const double &best_dist   //
    );

    // default caller
    KDNodePtr nearest_(const point_t &pt);

   public:
    point_t nearest_point(const point_t &pt);
    unsigned int nearest_index(const point_t &pt);
    pointIndex nearest_pointIndex(const point_t &pt);

   private:
    pointIndexArr neighborhood_(  //
        const KDNodePtr &branch,  //
        const point_t &pt,        //
        const double &rad,        //
        const unsigned int &level
    );
    pointIndexArr neighborhood_(  //
        const KDNodePtr &branch,  //
        const point_t &pt,        //
        const double &rad,        //
        const unsigned int &level,      //
        std::vector<double>&
    );

   public:
    pointIndexArr neighborhood(  //
        const point_t &pt,       //
        const double &rad);

    pointVec neighborhood_points(  //
        const point_t &pt,         //
        const double &rad);

    indexArr neighborhood_indices(  //
        const point_t &pt,          //
        std::vector<double> &,
        const double &rad);

    /* =============================================================
        User defined: find NNparticles fro certain group of
                      particles from a larger group of particles
    ============================================================== */
    indexArr neighborhood_indices_target( //
        const point_t &pt,                        //
        std::vector<double> &distPoint,
        const double &rad);
};
