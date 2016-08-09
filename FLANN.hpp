//
//  FLANN.hpp
//  FLANN
//
//  Created by Andrew Long on 5/23/16.
//  Copyright © 2016 Andrew Long. All rights reserved.
//

#ifndef FLANN_h
#define FLANN_h

#include <functional>
#include <string>
#include <utility>
#include <stdexcept>
#include <map>

#include "armadillo/include/armadillo.hpp"


#define ELEM_UNDEFINED -1

// Maintains a priority queue where the top priority item has a minimum score
template <typename T>
class MinHeap
{
private:
    typedef std::pair<double, T> PAIR;
    
    struct comparePair : public std::function<bool(T,T)>
    {
        bool operator() (const PAIR& lhs, const PAIR& rhs) const
        {
            return lhs.first > rhs.first;
        }
    };
    int count;
    std::vector<PAIR> heap;
    
public:
    MinHeap() { clear(); }
    ~MinHeap() { clear(); }
    int size() { return count; }
    bool empty() { return count == 0; }
    void clear() { heap.clear(); count = 0; }
    void push(double d, const T& obj)
    {
        heap.emplace_back(d, obj);
        std::push_heap(heap.begin(), heap.end(), comparePair());
        ++count;
    }
    bool pop()
    {
        if(count == 0)
            throw std::out_of_range("MinHeap<>::pop(): calling top on empty heap");
        std::pop_heap(heap.begin(), heap.end(), comparePair());
        heap.pop_back();
        --count;
        return true;
    }
    double top(T& obj)
    {
        if(count == 0)
            throw std::out_of_range("MinHeap<>::top(): calling top on empty heap");
        obj = heap[0].second;
        return heap[0].first;
    }
};



class SparseMatrix
{
    
public:
    
    typedef std::pair<int,int> loc_t;
    typedef std::map<loc_t,double> map_t;
    
    typedef std::map<loc_t,double>::iterator loc_itr;
    typedef std::pair<loc_itr,bool> map_itr;
    SparseMatrix()
    {
        clear();
    }
    
    void init(int size)
    {
        clear();
        N = size;
    }
    
    ~SparseMatrix()
    {
        clear();
    }
    
    void clear()
    {
        sp_mat.clear();
        N = 0;
    }
    
    loc_itr getElement(int i, int j)
    {
        checkInput(i,j);
        map_itr itr = sp_mat.insert(std::make_pair(std::make_pair(i,j),ELEM_UNDEFINED));
        return itr.first;
    }
    
    double operator()(int i, int j){
        checkInput(i,j);

        // grab iterator to position in map of item
        map_itr itr = sp_mat.insert(std::make_pair(std::make_pair(i,j), ELEM_UNDEFINED));
        // element already existed,
        if(itr.second == false)
            return itr.first->second;
        else
            return ELEM_UNDEFINED;
    }
    
    void insert(int i, int j, double d)
    {
        loc_itr itr = getElement(i,j);
        itr->second = d;
    }
    
private:
    
    map_t sp_mat;
    
    int N;
    void checkInput(int& i, int& j) const
    {
        if(i < 0 || i >= N || j < 0 || j >= N)
            throw std::out_of_range("SparseMatrix::(): index out of bounds");
        if(i > j)
        {
            std::swap(i, j);
        }
    }
};

// Sample K values from [0,N) without replacement
class RandSampler
{
private:
    arma::uvec m_numbers;
    int m_N;
public:
    RandSampler(int N)
    {
        m_N = N;
        m_numbers.set_size(m_N);
        
        for(int i = 0; i < N; ++i)
            m_numbers[i] = i;
    }
    void sampleK(int K, arma::uvec& samples)
    {
        if(K >= m_N)
        {
            samples.set_size(m_N);
            for(int i = 0; i < m_N; ++i)
                samples[i] = i;
        }
        else
        {
            int max = m_N;
            samples.set_size(K);
            for(int k = 0; k < K; k++)
            {
                // select new random index
                int idx = rand() % max;
                max--;
                // swap it to the current list end
                unsigned long val = m_numbers[idx];
                m_numbers[idx] = m_numbers[max];
                m_numbers[max] = val;
                // update the samples
                samples[k] = val;
            }
        }
    }
};

// generates a K-medoid Tree (akin to the K-means Tree of Lowe et al.)
// see http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_pami2014.pdf
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3003&rep=rep1&type=pdf
// original technique uses either
//      1) K-d trees (which require dimension additive distance measures)
//      2) K-means trees (which require that you can compute centers of a cluster via center of mass)
// instead use k-medoids to choose centers directly from the dataset, using these as branching points
template <typename T>
class KMedTree
{
public:
    struct KMedNode
    {
        KMedNode(int ID, int Nobj, int k, double v=1.0f) : id(ID), N(Nobj), K(k), var(v) {
            objs.set_size(N);
            meds.set_size(K);
        }
        ~KMedNode()
        {
            objs.clear();
            meds.clear();
            for(int i = 0 ; i < (int)children.size(); ++i)
            {
                KMedNode* tmp = children[i];
                delete tmp;
            }
        }
        int id; // the id of this object
        int N; // number of objects assigned to this node
        int K; // number of DIRECT children (branches) from this node
        bool isLeaf; // when N <= K
        double var;
        arma::uvec objs;
        arma::uvec meds;
        std::vector<KMedNode*> children;
        
    };
    
    KMedTree() : root(nullptr)
    {
        clear();
    }
    
    ~KMedTree()
    {
        clear();
    }
    
    void clear()
    {
        if(root != nullptr)
            delete root;
        root = nullptr;
        distances.clear();
        objects.clear();
    }
    
    void init(int nObj, int K, int itrMax, std::vector<T> obj, std::function<double(T,T)> dfunc);
    
    std::function<double(T,T)> _dist_func;
    
    
    int n_objects;
    int n_branch;
    int n_iters_max;
    int _dist_call_count;
    int _dist_call_query;
    void build();
    
    double getDistance(T& query, int v);
    double getDistance(int v1, int v2);
    // approximate M nearest neighbors to query, searching at most L points
    int queryTree_MNN(T& query, int M, int L, std::vector<int>& indexs, std::vector<double>& dists);
    int queryTree_interior_MNN(int sample, int M, int L, std::vector<int>& indexs, std::vector<double>& dists);
    
    int queryTree_radiusNN(T& query, double radius, int L, std::vector<int>& indexs, std::vector<double>& dists);
    int queryTree_interior_radiusNN(int sample, double radius, int L, std::vector<int>& indexs, std::vector<double>& dists);
    
    void predecessorList(arma::uvec& preds);
    
    void save(std::string directory);
    
private:
    SparseMatrix distances;
    std::vector<T> objects;
    KMedNode* root;
    
    void traverseTree(KMedNode* node, T& query, MinHeap<KMedNode*>& search_queue, MinHeap<int>& solution_queue, int&count, int L);
    void traverseTree_interior(KMedNode* node, int sample, MinHeap<KMedNode*>& search_queue, MinHeap<int>& solution_queue, int&count, int L);
    
    void buildNode(KMedNode* node);
    void assignToMeds(KMedNode* node, arma::sp_umat& selection);
    void updateMeds(KMedNode* node, arma::sp_umat& selection, arma::vec& variances);
    
    void recursePred(KMedNode* node, arma::uvec& preds);
    
};

template <typename T>
void KMedTree<T>::init(int nObj, int K, int itrMax, std::vector<T> obj, std::function<double(T,T)> dfunc)
{
    n_objects = nObj;
    n_branch = K;
    n_iters_max = itrMax;
    objects.assign(obj.begin(),obj.end());
    _dist_func = dfunc;
}

template <typename T>
void KMedTree<T>::build()
{
    _dist_call_count = 0;
    _dist_call_query = 0;
printf("Root Construction");
    distances.init(n_objects);
    
    root = new KMedNode(-1,n_objects, n_branch);
    for(unsigned long i = 0; i < n_objects; ++i)
    {
        root->objs[i] = i;
    }
    buildNode(root);
    printf("Construction finished after:%d distance computations\n",_dist_call_count);
}

template <typename T>
double KMedTree<T>::getDistance(T& query, int v)
{
    double d = _dist_func(query, objects[v]);
    ++_dist_call_query;
    return d;
}

template <typename T>
double KMedTree<T>::getDistance(int v1, int v2)
{
    if(v1 == v2)
        return 0;
    // using iterators to get/set the value inside the sparse matrix
    // inserting into a map returns the iterator pointing at the correct location
    SparseMatrix::loc_itr d_loc = distances.getElement(v1, v2);
    if(d_loc->second != ELEM_UNDEFINED)
        return d_loc->second;
    
    double d = _dist_func(objects[v1],objects[v2]);
    ++_dist_call_count;
    d_loc->second = d;

    return d;
}

template <typename T>
void KMedTree<T>::assignToMeds(KMedNode* node, arma::sp_umat& selection)
{
    selection = arma::sp_umat(node->N, node->K);
    for(int i = 0; i < node->N; ++i)
    {
        int v1 = (int)node->objs[i];
        int min_idx = -1;
        double min_dist = arma::datum::inf;
        for(int j = 0; j < node->K; ++j)
        {
            int v2 = (int)node->objs[node->meds[j]];
            if(v1 == v2)
            {
                min_idx = j;
                break;
            }
            else
            {
                double d = getDistance(v1,v2);
                if( d < min_dist )
                {
                    min_dist = d;
                    min_idx = j;
                }
            }
        }
        selection(i,min_idx) = 1;
    }
}

template <typename T>
void KMedTree<T>::updateMeds(KMedNode* node, arma::sp_umat& selection, arma::vec& vars)
{
    arma::uvec set;
    for(int k = 0; k < node->K; ++k)
    {
        set = arma::find(arma::uvec(selection.col(k)));
        // find most central point between them all
        int min_idx = -1;
        double min_sum_dist = arma::datum::inf;
        double min_var = arma::datum::inf;
        for(int i = 0; i < set.n_elem; ++i)
        {
            double sum_dist = 0.f;
            double var = 0.f;
            for(int j = 0; j < set.n_elem; ++j)
            {
                if(i == j)
                    continue;
                
                int v1 = (int)node->objs[set[i]];
                int v2 = (int)node->objs[set[j]];
                
                double d = getDistance(v1,v2);
                sum_dist += d;
                var += d*d;
            }
            if(sum_dist < min_sum_dist)
            {
                min_sum_dist = sum_dist;
                min_idx = i;
                min_var = var;
            }
        }
        node->meds[k] = set[min_idx];
        vars[k] = min_var/set.n_elem;
    }
}

// performs KMed on the node object
template <typename T>
void KMedTree<T>::buildNode(KMedNode* node)
{
    if(node->N <= node->K)
    {
        node->isLeaf = true;
        for(int i = 0 ; i < node->N; ++i)
        {
            node->meds[i] = node->objs[i];
        }
        return;
    }
    node->isLeaf = false;
    
    // choose random set of medoids
    RandSampler rSampler(node->N);
    rSampler.sampleK(node->K, node->meds);
    
    node->meds = arma::sort(node->meds);
    arma::sp_umat selection;
    arma::uvec old_meds = node->meds;
    arma::uvec diff;
    arma::vec vars(node->K);
    arma::uvec sindex;
    bool converged = false;
    ;
    for(int i = 0; i < n_iters_max && !converged; ++i)
    {
        // assign to medoids
        assignToMeds(node,selection);
        
        // select new medoids
        updateMeds(node, selection, vars);
        
        // update and sort medoids based on numerical order
        //NOTE(AWL): probably unnecessary?
        sindex = arma::sort_index(node->meds);
        node->meds = node->meds(sindex);
        vars = vars(sindex);
        
        // determine if converged
        diff = arma::find(old_meds != node->meds,1);
        if(diff.is_empty())
            converged = true;
        else
            old_meds = node->meds;
    }
    // build sub nodes
    arma::uvec tmp;
    arma::uvec tmp_objs;
    for(int k = 0; k < node->K; ++k)
    {
        int tmp_id = (int)node->objs[node->meds[k]];
        
        tmp = arma::uvec(selection.col(k));
        tmp_objs = arma::find(tmp);
        int tmp_nelem = (int)tmp_objs.n_elem;

        KMedNode* newNode = new KMedNode(tmp_id,tmp_nelem, n_branch, vars[k]);
        newNode->objs = node->objs(tmp_objs);
        
        node->children.push_back(newNode);
    }
    
    // no longer need to worry about these variables once you've assembled the child nodes
    node->objs.clear();
    for(int k = 0; k < node->K; ++k)
    {
        buildNode(node->children[k]);
    }
    
}

template <typename T>
int KMedTree<T>::queryTree_MNN(T& query, int M, int L, std::vector<int>& indexs, std::vector<double>& dists)
{
    int count = 0;
    // evaluate the top L candidates
    MinHeap<KMedNode*> search_queue;
    MinHeap<int> solution_queue;
    traverseTree(root, query, search_queue, solution_queue, count, L);
    KMedNode* node;
    while( !search_queue.empty() && count < L)
    {
        if(!search_queue.top(node))
            break;
        search_queue.pop();
        traverseTree(node, query, search_queue, solution_queue, count, L);
    }
    
    // select the top M candidates
    int num_neigh  = std::min(M, solution_queue.size());
    indexs.clear(); indexs.reserve(num_neigh);
    dists.clear(); dists.reserve(num_neigh);
    
    for(int i = 0; i < num_neigh; ++i)
    {
        int x;
        double d;
        d = solution_queue.top(x); solution_queue.pop();
        
        if(d == -1) printf("Error: solution queue pushed past end!");
        indexs.push_back(x);
        dists.push_back(d);
        
    }
    return num_neigh;
}

template <typename T>
int KMedTree<T>::queryTree_interior_MNN(int sample, int M, int L, std::vector<int>& indexs, std::vector<double>& dists)
{
    int count = 0;
    // evaluate the top L candidates
    MinHeap<KMedNode*> search_queue;
    MinHeap<int> solution_queue;
    traverseTree_interior(root, sample, search_queue, solution_queue, count, L);
    KMedNode* node;
    while( !search_queue.empty() && count < L)
    {
        if(!search_queue.top(node))
            break;
        search_queue.pop();
        traverseTree_interior(node, sample, search_queue, solution_queue, count, L);
    }
    
    // select the top M candidates
    int num_neigh  = std::min(M, solution_queue.size());
    indexs.clear(); indexs.reserve(num_neigh);
    dists.clear(); dists.reserve(num_neigh);
    
    for(int i = 0; i < num_neigh; ++i)
    {
        int x;
        double d;
        d = solution_queue.top(x); solution_queue.pop();
        
        if(d == -1) printf("Error: solution queue pushed past end!");
        indexs.push_back(x);
        dists.push_back(d);
        
    }
    return num_neigh;
}

template <typename T>
int KMedTree<T>::queryTree_radiusNN(T& query, double radius, int L, std::vector<int>& indexs, std::vector<double>& dists)
{
    int count = 0;
    // evaluate the top L candidates
    MinHeap<KMedNode*> search_queue;
    MinHeap<int> solution_queue;
    traverseTree(root, query, search_queue, solution_queue, count, L);
    KMedNode* node;
    while( !search_queue.empty() && count < L)
    {
        if(!search_queue.top(node))
            break;
        search_queue.pop();
        traverseTree(node, query, search_queue, solution_queue, count, L);
    }
    
    // select all candidates within
    int num_neigh = solution_queue.size();
    indexs.clear(); indexs.reserve(num_neigh);
    dists.clear(); dists.reserve(num_neigh);
    count = 0;
    for(int i = 0; i < num_neigh; ++i)
    {
        int x;
        double d;
        d = solution_queue.top(x); solution_queue.pop();
        
        if(d == -1) printf("Error: solution queue pushed past end!");
        if(d > radius) break;
        
        indexs.push_back(x);
        dists.push_back(d);
        ++count;
        
    }
    return count;
}

template <typename T>
int KMedTree<T>::queryTree_interior_radiusNN(int sample, double radius, int L, std::vector<int>& indexs, std::vector<double>& dists)
{
    int count = 0;
    // evaluate the top L candidates
    MinHeap<KMedNode*> search_queue;
    MinHeap<int> solution_queue;
    traverseTree_interior(root, sample, search_queue, solution_queue, count, L);
    KMedNode* node;
    while( !search_queue.empty() && count < L)
    {
        if(!search_queue.top(node))
            break;
        search_queue.pop();
        traverseTree_interior(node, sample, search_queue, solution_queue, count, L);
    }
    
    // select all candidates within
    int num_neigh = solution_queue.size();
    indexs.clear(); indexs.reserve(num_neigh);
    dists.clear(); dists.reserve(num_neigh);
    count = 0;
    for(int i = 0; i < num_neigh; ++i)
    {
        int x;
        double d;
        d = solution_queue.top(x); solution_queue.pop();
        
        if(d == -1) printf("Error: solution queue pushed past end!");
        if(d > radius) break;
        
        indexs.push_back(x);
        dists.push_back(d);
        ++count;
        
    }
    return count;
}

template <typename T>
void KMedTree<T>::traverseTree(KMedNode* node, T& query, MinHeap<KMedNode*>& search_queue, MinHeap<int>& solution_queue, int&count, int L)
{  
    if(node->isLeaf)
    {
        
        for(int i = 0; i < node->N; ++i)
        {
            double dist = getDistance(query, node->meds[i]); //_dist_func(query, objects[node->meds[i]]);
            solution_queue.push(dist, (int)(node->meds[i]));
        }
        count += node->N;
    }
    else
    {
        arma::uvec dists(node->K);
        int min_idx = -1;
        double min_dist = arma::datum::inf;
        for(int i = 0; i < node->K; ++i)
        {
            dists(i) = getDistance(query, node->meds[i]);//_dist_func(query, objects[node->meds[i]]);
            if(dists(i) < min_dist)
            {
                min_idx = i;
                min_dist = dists(i);
            }
        }
        for(int i = 0; i < node->K; ++i)
        {
            if(i == min_idx)
                continue;
            double domain = dists(i) - 0.4*node->children[i]->var;
            search_queue.push(domain, node->children[i]);
        }
        traverseTree(node->children[min_idx], query, search_queue, solution_queue, count, L);
    }
}

template <typename T>
void KMedTree<T>::traverseTree_interior(KMedNode* node, int sample, MinHeap<KMedNode*>& search_queue, MinHeap<int>& solution_queue, int&count, int L)
{
    if(node->isLeaf)
    {
        // push all of the leaf nodes onto the PQueue
        for(int i = 0; i < node->N; ++i)
        {
            double dist = getDistance(sample, (int)(node->meds[i]));
            solution_queue.push(dist, (int)(node->meds[i]));
        }
        count += node->N;
    }
    else
    {
        // recurse down the tree by selecting the closest medoid branch
        arma::vec dists(node->K);
        int min_idx = -1;
        double min_dist = arma::datum::inf;
        for(int i = 0; i < node->K; ++i)
        {
            dists(i) = getDistance(sample, (int)(node->meds[i]));
            if(dists(i) < min_dist)
            {
                min_idx = i;
                min_dist = dists(i);
            }
        }
        for(int i = 0; i < node->K; ++i)
        {
            if(i == min_idx)
                continue;
            double domain = dists(i) - 0.4*node->children[i]->var;
            search_queue.push(domain, node->children[i]);
        }

        traverseTree_interior(node->children[min_idx], sample, search_queue, solution_queue, count, L);
    }
}


template<typename T>
void KMedTree<T>::predecessorList(arma::uvec& preds)
{
    preds.set_size(n_objects);
    for(int i = 0; i < root->K; ++i)
    {
        preds[root->children[i]->id] = -1;
        recursePred(root->children[i], preds);
    }
}

template<typename T>
void KMedTree<T>::recursePred(KMedNode* node, arma::uvec& preds)
{
    if(node->isLeaf)
    {
        for(int i = 0; i < node->N; ++i)
        {
            if(node->meds[i] != node->id)
                preds[node->meds[i]] = node->id;
        }
        return;
    }
    
    for(int i = 0; i < node->K; ++i)
    {
        if(node->children[i]->id != node->id)
            preds[node->children[i]->id] = node->id;
        recursePred(node->children[i], preds);
    }
}



#endif /* FLANN_h */
