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
    foreheadWidthRatio  = 0.20;
    foreheadHeightRatio = 0.125;
    maxMeasurement = 50;
    FPS = 0.0;
    
    minBPM = 50.0;
    maxBPM = 180.0;
    
    /* Slices of smoked hams between toasted buns, with mayo, ketchup, caramelized onion, pickles, and harissa hot sauce */
    vector<double> ham_ham_ham_ham_ham_ham = HeartBeat::hammingWindow(maxMeasurement);
    hamming = Mat(ham_ham_ham_ham_ham_ham);
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
    vector<double> freq         = calcFrequency( maxMeasurement, FPS, 60.0 );
    vector<double> even_time    = linspace( measure_time[0], measure_time[maxMeasurement - 1], maxMeasurement );
    vector<double> interpolated = interp( even_time, time_vec, measure_vec );
    Mat interpolated_mat( interpolated );
    
    interpolated_mat = hamming.mul( interpolated_mat );
    interpolated_mat = interpolated_mat - cv::mean( interpolated_mat );
    
    Mat fft_output;
    dft( interpolated_mat, fft_output, DFT_COMPLEX_OUTPUT );
    
    vector<Mat> components;
    split(fft_output, components);
    
    Mat fft_abs;
    magnitude( components[0], components[1], fft_abs );
    
    Mat freq_mask( freq );
    freq_mask = (freq_mask > minBPM & freq_mask < maxBPM );
    
    Mat pruned;
    fft_abs.copyTo( pruned, freq_mask );

    auto res = max_element( pruned.begin<double>(), pruned.end<double>());
    size_t max_index = distance( pruned.begin<double>(), res );
    
    return freq[max_index];
}

Rect HeartBeat::getForeheadRegion( Rect& face, float fh_x, float fh_y, float fh_w, float fh_h ) {
    return Rect( face.x + face.width  * fh_x - (face.width  * fh_w / 2.0),
                 face.y + face.height * fh_y - (face.height * fh_h / 2.0),
                 face.width  * fh_w, face.height * fh_h );
}

#define STATE_INITIAL       0
#define STATE_DETECT_FACE   1
#define STATE_FACE_DETECTED 2

void HeartBeat::run() {
    namedWindow( "" );
    moveWindow("", 0, 0);
    
    VideoCapture cap(0);
    Mat frame, prev_frame, gray;
    
    double start = (chrono::system_clock::now().time_since_epoch() / chrono::microseconds(1)) / 1000000.0;
    CvFont font = cvFontQt("Helvetica", 18.0, CV_RGB(0, 255, 0) );
    char temp[255];
    
    Rect best_face;
    int state = STATE_INITIAL;
    
    Mat status, error;
    
    while( true ) {
        cap >> frame;
        flip( frame, frame, 1 );
        
        cvtColor( frame, gray, CV_BGR2GRAY );
        equalizeHist( gray, gray );
        
        if( state != STATE_FACE_DETECTED ){
            vector<Rect> faces;
            faceClassifier.detectMultiScale( frame, faces );
            
            if( faces.empty() == false ) {
                stable_sort( faces.begin(), faces.end(), [&](Rect a, Rect b) {
                    return a.area() < b.area();
                });
                
                
                rectangle(frame, faces[0], CV_RGB(255, 0, 0));
                
                if( state == STATE_DETECT_FACE ){
                    best_face = faces[0];
                    state = STATE_FACE_DETECTED;
                }
            }
        }
        
        if( state == STATE_FACE_DETECTED ) {
            rectangle(frame, best_face, CV_RGB(255, 0, 0));
            
            
            Rect forehead = getForeheadRegion( best_face, 0.5, 0.18, 0.25, 0.15 );
            Mat subregion = Mat(frame, forehead);
            Scalar mean_values = cv::mean( subregion );
            double average = mean_values[1];
            
            rectangle(frame, forehead, CV_RGB(0, 255, 0));
            
            double now = (chrono::system_clock::now().time_since_epoch() / chrono::microseconds(1)) / 1000000.0;
            addSample( now - start, average );
            
            double bpm = getBPM( measurement, measurementTime );

            sprintf( temp, "BPM: %.2f", bpm );
            addText( frame, temp, Point(forehead.x, forehead.y), font );
        }
        
        
        
        
        imshow( "", frame );
        char key = waitKey(10);
        if( key == 's')
            state = STATE_DETECT_FACE;
        else if( key == 'q' )
            break;
        
        prev_frame = frame.clone();
    }
}

/**
 * Create a vector of values that are evenly spaced between start and end values
 * A mimic of numpy.linspace
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
 * http://en.wikipedia.org/wiki/Linear_interpolation 
 * even when the formula is simple, when it's converted to C++, it's hard to read
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


/**
 * Hamming window for use with FFT as window function 
 * http://en.wikipedia.org/wiki/Window_function#Hamming_window
 * Read more here:
 * http://stackoverflow.com/questions/7337709/why-do-i-need-to-apply-a-window-function-to-samples-when-building-a-power-spectr
 */
vector<double> HeartBeat::hammingWindow( int n ){
    vector<double> res(n);
    for( int i = 0; i < n; i++ )
        res[i] = 0.54 - 0.46 * cos( (2 * M_PI * i) / (n-1)  );
    return res;
}

/**
 * Calculate the frequency
 */
vector<double> HeartBeat::calcFrequency( int n, double fps, double scale ) {
    int size = n / 2 + 1;
    vector<double> result( n, 0 );
    for( int i = 1; i < size; i++ )
        result[i] = scale * fps / n * i;
    return result;
}