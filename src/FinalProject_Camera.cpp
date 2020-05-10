
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

struct perfStats {
  string detectorType;
  string descriptorType;
  string matcherType;
  string selectorType;
  int   numKeyPointsPerframe[20];
  int   numKeyPointsPerROI[20];
  int   numMatchedKeyPoints[20];
  float neighboorhoodSizeMean[20];
  float neighboorhoodSizeVariance[20];
  float matchDistanceMean[20];
  float matchDistanceVariance[20];
  double detectorTime[20];
  double descriptorTime[20];
  double matcherTime[20];
  double TTC[20];
};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    // struct to hold performances for evalutation 
    perfStats performances;

    std::string filename = "../data.csv";
    std::ofstream output_stream(filename, std::ios::binary | std::ios::app);

    if (!output_stream.is_open()) {
        std::cerr << "failed to open file: " << filename << std::endl;
        return EXIT_FAILURE;
    }

    vector<string> detector_list   = {"SHITOMASI","HARRIS","FAST","BRISK","ORB","AKAZE","SIFT"};
    vector<string> descriptor_list = {"BRISK","BRIEF","ORB","FREAK","AKAZE","SIFT"};
    string detector, descriptor;

    for (int det = 0; det < detector_list.size() ; det++) // detector loop
    {
        detector = detector_list[det];
        cout << "Detector is " << detector << endl;

    for (int des = 0; des < descriptor_list.size(); des++) // descriptor loop
    {
        descriptor = descriptor_list[des];
        cout << "Descriptor is " << descriptor << endl;

    if ( descriptor.compare("AKAZE") == 0 && detector.compare("AKAZE") != 0 )
    {
        // AKAZE descriptor requires AKAZE detector only.
        continue;
    }
    if ( detector.compare("SIFT") == 0 && descriptor.compare("ORB") == 0 )
    {
        // SIFT detector and ORB descriptor combination not valid.
        continue;
    }
    while (!dataBuffer.empty())
    {
        dataBuffer.erase(dataBuffer.begin());
    }

    // write CSV header row
    output_stream << "Detector Type" << ","
                << "Descriptor Type" << ","
                << "Frame#" << ","
                << "#KeyPointsPerFrame" << ","
                << "#KeyPointsPerROI" << ","
                << "NeighborhoodSizeMean" << ","
                << "NeighborhoodSizeVariance" << ","
                << "DetectorTime(ms)" << ","
                << "DescriptorTime(ms)" << ","
                << "Matcher Type" << ","
                << "Selector Type" << ","
                << "#MatchedPoints" << "," 
                << "MatchDistancesMean" << ","
                << "MatchDistancesVariance" << ","
                << "MatchingTime(ms))" << ","
                << "TTC(s))" << std::endl;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */
        
        bVis = false;
        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
        bVis=false;

        // cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        // cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        // cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        // continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = detector; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        performances.detectorType = detectorType;

        if (detectorType.compare("SHITOMASI") == 0)
        {
            performances.detectorTime[imgIndex] = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if(detectorType.compare("HARRIS") == 0)
        {
            performances.detectorTime[imgIndex] = detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0 || detectorType.compare("BRISK") == 0 || detectorType.compare("ORB") == 0 || detectorType.compare("AKAZE") == 0 || detectorType.compare("SIFT") == 0)
        {
            performances.detectorTime[imgIndex] = detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        {
            cout << "The detector type is not implemented. Exiting now." << endl;
            return EXIT_FAILURE;
        }

        performances.numKeyPointsPerframe[imgIndex] = keypoints.size();

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        // cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        string descriptorType = descriptor; // argv[2]; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        performances.descriptorType = descriptorType;
        performances.descriptorTime[imgIndex] = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        if (performances.descriptorTime[imgIndex] < 0)
        {
            cout << "Descriptor extraction failed." << endl;
            return EXIT_FAILURE;
        }  

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            performances.matcherType = matcherType;
            //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
            performances.selectorType = selectorType;    

            performances.matcherTime[imgIndex] = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            performances.numMatchedKeyPoints[imgIndex] = matches.size();

            // Evaluation of quality of matches. 
            // 1.- Statistics of match distance metric
            vector<float> distances(matches.size());
            for (auto match = matches.begin(); match != matches.end(); match++)
            {
                distances.push_back((*match).distance);
            }
            // Mean of distances
            float sum = std::accumulate(distances.begin(), distances.end(), 0.0);
            float distances_mean = sum / distances.size();

            // Standard Dev of distances
            vector<float> diff(distances.size());
            transform(distances.begin(), distances.end(), diff.begin(), [distances_mean](float x) { return x - distances_mean; });
            float sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            float distances_variance = sq_sum / distances.size();

            performances.matchDistanceMean[imgIndex] = distances_mean;
            performances.matchDistanceVariance[imgIndex] = distances_variance;

            // 2.- Statistics of keypoints euclidean distance in image
            vector<float> distances2(matches.size());
            for (auto match = matches.begin(); match != matches.end(); match++)
            {
                cv::KeyPoint queryKpt = (dataBuffer.end() - 2)->keypoints[(*match).queryIdx];
                cv::KeyPoint trainKpt = (dataBuffer.end() - 1)->keypoints[(*match).trainIdx];
                distances2.push_back(norm(queryKpt.pt-trainKpt.pt));
            }
            // Mean of distances
            float sum2 = std::accumulate(distances2.begin(), distances2.end(), 0.0);
            float mean2 = sum2 / distances2.size();

            // Standard Dev of distances
            vector<float> diff2(distances2.size());
            transform(distances2.begin(), distances2.end(), diff2.begin(), [mean2](float x) { return x - mean2; });
            float sq_sum2 = inner_product(diff2.begin(), diff2.end(), diff2.begin(), 0.0);
            float stdev2 = sqrt(sq_sum2 / distances2.size());

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // showing contents of bbBestMatches
            // std::cout << "bbBestMatches contains:\n";
            // for (auto it=bbBestMatches.begin(); it!=bbBestMatches.end(); ++it)
            //    std::cout << it->first << " => " << it->second << '\n';


            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            // cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    performances.TTC[imgIndex] = ttcCamera;

                    performances.numKeyPointsPerROI[imgIndex] = currBB->keypoints.size();

                    // Evaluate distribution of neighborhood size of keypoints
                    vector<float> sizes(currBB->keypoints.size());
                    for (auto keypoint = currBB->keypoints.begin(); keypoint != currBB->keypoints.end(); keypoint++)
                    {
                        sizes.push_back((*keypoint).size);
                    } 
                    double sizes_mean = std::accumulate(sizes.begin(), sizes.end(), 0.0)/sizes.size();
                    auto add_square = [sizes_mean](double sum, int i)
                    {
                        auto d = i - sizes_mean;
                        return sum + d*d;
                    };
                    double total = std::accumulate(sizes.begin(), sizes.end(), 0.0, add_square);
                    double sizes_variance = total / sizes.size();

                    performances.neighboorhoodSizeMean[imgIndex] = sizes_mean;
                    performances.neighboorhoodSizeVariance[imgIndex] = sizes_variance;


                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches            

        }
        else
        {
            performances.matcherType = "MAT_BF"; // argv[3];
            performances.selectorType = "SEL_KNN"; // argv[4];
            performances.matcherTime[imgIndex] = 0.0;
            performances.numMatchedKeyPoints[imgIndex] = 0;
            performances.numKeyPointsPerROI[imgIndex] = 0;
            performances.neighboorhoodSizeMean[imgIndex] = 0.0;
            performances.neighboorhoodSizeVariance[imgIndex] = 0.0;
            performances.TTC[imgIndex] = 0.0;
            cout << endl;
        }
        

    } // eof loop over all images
    for (int i = 0; i < 10; i++) 
    {
        output_stream << performances.detectorType
                << "," << performances.descriptorType
                << "," << i
                << "," << performances.numKeyPointsPerframe[i]
                << "," << performances.numKeyPointsPerROI[i]
                << "," << performances.neighboorhoodSizeMean[i]
                << "," << performances.neighboorhoodSizeVariance[i]
                << "," << std::fixed << std::setprecision(3) << performances.detectorTime[i]
                << "," << std::fixed << std::setprecision(3) << performances.descriptorTime[i]
                << "," << performances.matcherType
                << "," << performances.selectorType
                << "," << performances.numMatchedKeyPoints[i]
                << "," << performances.matchDistanceMean[i]
                << "," << performances.matchDistanceVariance[i]
                << "," << std::fixed << std::setprecision(3) << performances.matcherTime[i]
                << "," << std::fixed << std::setprecision(3) << performances.TTC[i] << std::endl;
    }
    output_stream << std::endl;
    } // end loop descriptors
    } // end loop detectors
    output_stream.close();

    return 0;
}
