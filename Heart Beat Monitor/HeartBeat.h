//
//  HeartBeat.h
//  Heart Beat Monitor
//
//  Created by Saburo Okita on 10/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#ifndef __Heart_Beat_Monitor__HeartBeat__
#define __Heart_Beat_Monitor__HeartBeat__

#include <iostream>
#include <ctime>
#include <deque>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


template <typename type>
static ostream& operator<<(ostream&os, vector<type>& vec) {
    os << "[";
    for( type v : vec )
        os << v << ", ";
    os << "]";
    return os;
}


class HeartBeat {
protected:
    int maxMeasurement;
    deque<double> measurementTime;
    deque<double> measurement;
    float FPS;
    float foreheadHeightRatio;
    float foreheadWidthRatio;
    float maxBPM;
    float minBPM;
    float heartrate;
    Mat hamming;
    CascadeClassifier faceClassifier;

    
public:
    HeartBeat();
    double getBPM( deque<double>& measurement, deque<double>& measurementTime );
    bool enoughMeasurement();
    
    void run();
    Rect getForeheadRegion( Rect& face, float foreheadWidthRatio, float foreheadHeightRatio );
    void addSample( double time_measurement, double average );
    
    template <typename type>
    static vector<type> linspace( type start, type end, int length );
    
    template <typename type>
    static vector<type> interp( vector<type>& x, vector<type>& xp, vector<type>& yp );
    static vector<double> hammingWindow( int n );
    static vector<double> calcFrequency( int n, double fps, double scale );
};


#endif /* defined(__Heart_Beat_Monitor__HeartBeat__) */
