//
//  main.cpp
//  Heart Beat Monitor
//
//  Created by Saburo Okita on 07/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <ctime>

#include "HeartBeat.h"

using namespace std;
using namespace cv;


int main(int argc, const char * argv[])
{
    
    HeartBeat monitor;
    monitor.run();

    return 0;
}
