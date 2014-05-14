//
//  HeartBeat.cpp
//  Heart Beat Monitor
//
//  Created by Saburo Okita on 10/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include "HeartBeat.h"
#include <chrono>

HeartBeat::HeartBeat() {
    faceClassifier.load( "/usr/local/Cellar/opencv/2.4.8.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml" );
    foreheadWidthRatio  = 0.25;
    foreheadHeightRatio = 0.125;
    maxMeasurement = 10;
    FPS = 0.0;
    
    minBPM = 50.0;
    maxBPM = 200.0;
    
    
    vector<double> ham = HeartBeat::hammingWindow(maxMeasurement);
    hamming = Mat(ham);
}

void HeartBeat::addSample( double time_measurement, double average ) {
    if( measurementTime.size() >= maxMeasurement )
        measurementTime.pop_front();
    measurementTime.push_back( time_measurement );
    
    if( measurement.size() >= maxMeasurement )
        measurement.pop_front();
    measurement.push_back( static_cast<double>( average ) );
}



bool HeartBeat::enoughMeasurement() {
    return measurement.size() >= maxMeasurement;
}

double HeartBeat::getBPM( deque<double>& measure, deque<double>& measure_time ) {
    vector<double> measure_vec( measure.begin(), measure.end() );
    vector<double> time_vec( measure_time.begin(), measure_time.end() );
    
    FPS = (1.0 * maxMeasurement) / ( time_vec[maxMeasurement-1] - time_vec[0] );
    
    vector<double> even_time      = linspace( measure_time[0], measure_time[maxMeasurement - 1], maxMeasurement );
    Mat interpolated              = Mat( HeartBeat::interp( even_time, time_vec, measure_vec ) );
    
    interpolated = interpolated.mul( hamming );
    interpolated = interpolated - cv::mean( interpolated );
    
    Mat fft_output;
    dft( interpolated, fft_output, DFT_COMPLEX_OUTPUT );
    
    vector<Mat> components;
    split(fft_output, components);
    
    Mat fft_abs, fft_phase;
    magnitude( components[0], components[1], fft_abs );
    cv::phase( components[0], components[1], fft_phase );
    
    vector<double> freq = calcFrequency( maxMeasurement, FPS, 60.0 );
    
    Mat freq_mask( freq );
    freq_mask = (freq_mask > 50 & freq_mask < 180);
    Mat pruned;
    fft_abs.copyTo( pruned, freq_mask );

    auto res = max_element( pruned.begin<double>(), pruned.end<double>());
    size_t max_index = distance( pruned.begin<double>(), res );
    
    return freq[max_index];
}

Rect HeartBeat::getForeheadRegion( Rect& face, float w_ratio, float h_ratio ) {
     return Rect( face.x + (face.width * 0.5) * (1.0 - w_ratio),
                  face.y + (face.height * 0.125) * (1.0 - h_ratio),
                  face.width * w_ratio,
                  face.height * h_ratio );
}

void HeartBeat::run() {
    VideoCapture cap(0);
    Mat frame;
    
    double start = (chrono::system_clock::now().time_since_epoch() / chrono::microseconds(1)) / 1000000.0;
    
    CvFont font = cvFontQt("Helvetica", 18.0, CV_RGB(0, 255, 0) );
    char temp[255];
    
    vector<Rect> faces;
    
    
    while( true ) {
        cap >> frame;
        flip( frame, frame, 1 );
        
        double now = (chrono::system_clock::now().time_since_epoch() / chrono::microseconds(1)) / 1000000.0;
        
        faceClassifier.detectMultiScale( frame, faces, CV_HAAR_FIND_BIGGEST_OBJECT );
        
        if( !faces.empty() ){
            rectangle(frame, faces[0], Scalar(0, 0, 255 ));
            
            Rect forehead = getForeheadRegion( faces[0], foreheadWidthRatio, foreheadHeightRatio );
            rectangle(frame, forehead, Scalar(0, 255, 0 ));
            
            Mat subregion = Mat(frame, forehead);

            Scalar avg = cv::sum( cv::mean(subregion) ) / 3.0;
            addSample( now - start , avg[0] );
        
            
            if( enoughMeasurement() ){
                double bpm = getBPM( measurement, measurementTime );
                sprintf( temp, "BPM: %f", bpm );
                addText( frame, temp, Point( faces[0].x + 10, faces[0].y), font );
            }
        }
        
        
        imshow( "", frame );
        if( waitKey(10) == 'q' )
            break;
    }
}

/**
 * Create a vector of values that are evenly spaced between start and end values
 */
template <typename type>
vector<type> HeartBeat::linspace( type start, type end, int length ) {
    vector<type> result;
    type step = (end - start) / (length - 1);
    type value = start;
    
    for( int i = 1; i < length; i++ ){
        result.push_back( value );
        value += step;
    }
    
    result.push_back( end );
    return result;
}

/**
 * Perform linear interpolation
 */
template <typename type>
vector<type> HeartBeat::interp( vector<type>& x, vector<type>& xp, vector<type>& yp ) {
    vector<type> y;
    
    size_t size = x.size() - 1;
    for( int i = 0; i < size; i++ )
        y.push_back( yp[i] + (yp[i+1] - yp[i]) * ((x[i] - xp[i]) / (xp[i+1] - xp[i])) );
    
    y.push_back( yp[size] );
    
    return y;
}



vector<double> HeartBeat::hammingWindow( int n ){
    vector<double> res(n);
    for( int i = 0; i < n; i++ )
        res[i] = 0.54 - 0.46 * cos( (2 * M_PI * i) / (n-1)  );
    return res;
}


vector<double> HeartBeat::calcFrequency( int n, double fps, double scale ) {
    int size = n / 2 + 1;
    vector<double> result( n, 0 );
    for( int i = 1; i < size; i++ )
        result[i] = scale * fps / n * i;
    return result;
}