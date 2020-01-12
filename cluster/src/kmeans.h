#ifndef KMEANS_H
#define KMEANS_H
#include <vector>
#include <iostream>
#include <set>
#include "Curve.h"
#include "distance.h"
#include "Point.h"

class Kmeans {
    protected:
    set<Object *> objs;
    Object *currCentroid;
    Object *prevCentroid;
    DistanceMetric *metric;
    double stopThreshold;
    public:
    Kmeans(double stopThreshold) {
        this->stopThreshold = stopThreshold;
        this->currCentroid = nullptr;
        this->prevCentroid = nullptr;
    }
    void setObjs(set<Object *> objs) {
        this->objs = objs;
    }
    bool canStop() {
        if (prevCentroid == nullptr)
            return false;
        //cout << "DIST:" << metric->dist(currCentroid,prevCentroid) << endl;
        if (metric->dist(currCentroid,prevCentroid) < stopThreshold) 
            return true;
        return false;
    }
    double getThreshold() { return stopThreshold;}
    virtual Object *centroid(bool *stop) = 0;

};

class KmeansPoints : public Kmeans {
    int numDimension;
    public:
        KmeansPoints(double stopThreshold,int numDimension) : 
        Kmeans(stopThreshold) {
            this->numDimension = numDimension;
            this->metric = new Manhattan();
        };
        Object * centroid(bool *stop) {
            double size = double(objs.size());
            vector<double> centro(numDimension,0.0);
            prevCentroid = currCentroid;
            for (auto obj : objs) {
                auto p = dynamic_cast<Point *>(obj);
                for (int i =0; i < p->getCoordinates().size();i++) {
                    centro.at(i) += p->getCoordinate(i)/size;
                }
            }
            currCentroid = new Point(centro);
            *stop = canStop();
            return currCentroid; 
        }

};


struct point_compare {
   bool operator() (const Point p1,const Point p2) {
       auto point1 = p1;
       auto point2 = p2;
    if (point1.getCoordinates().size() !=  point2.getCoordinates().size())
        return true;
    for (int i =0; i<point2.getCoordinates().size(); i++) 
        if (point1.getCoordinate(i) == point2.getCoordinate(i))
            continue;
        else if (point1.getCoordinate(i) <point2.getCoordinate(i))
            return true;
        else 
            return false;
    return false;
   }
};


class DBA : public Kmeans {
    //array of point sets
    int centroidLen;
    //algorithm ends if dist(nextCentroid,currCentroid) < endDistThreshold
    int meanLength();
    Point mean(set<Point,point_compare> pset) ;
    Curve *pickRandomFilterShort();
    Curve *pickRandomSubsequence(Curve *curve);
    public:
        DBA(double stopThreshold);
        Object* centroid(bool *stop);
};

#endif