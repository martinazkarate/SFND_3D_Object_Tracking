
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (auto match = kptMatches.begin(); match != kptMatches.end(); match++) // Iterate over the vector of matches
    {
        if (boundingBox.roi.contains(kptsCurr[match->trainIdx].pt))
            boundingBox.kptMatches.push_back(*match);
    }

    if (boundingBox.kptMatches.size() == 0)
    {
        std::cerr << "No keypoints found within the Bounding Box" << std::endl;
        return;
    }

    vector<float> distances(boundingBox.kptMatches.size());
    for (auto match = kptMatches.begin(); match != kptMatches.end(); match++)
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
    
    for (auto match = kptMatches.begin(); match != kptMatches.end(); match++)
    {
        if (match->distance > (distances_mean+sqrt(distances_variance)))
        {
            kptMatches.erase(match);
            match--;
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ransac parameters for segmentation
    int maxIterations = 50;
    float distanceTol = 0.2;

    // Segment Plane from previous lidar points
    segmentPlane(lidarPointsPrev, maxIterations, distanceTol);
    
    // Segment Plane from current lidar points
    segmentPlane(lidarPointsCurr, maxIterations, distanceTol);

    float prev_xmin = 1e8;
    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1)
    {
        float x = (*it1).x; // world position in m with x facing forward from sensor
        prev_xmin = prev_xmin<x ? prev_xmin : x;
    }

    float curr_xmin = 1e8;
    for (auto it2 = lidarPointsCurr.begin(); it2 != lidarPointsCurr.end(); ++it2)
    {
        float x = (*it2).x; // world position in m with x facing forward from sensor
        curr_xmin = curr_xmin<x ? curr_xmin : x;
    }

    TTC = curr_xmin / frameRate / (prev_xmin - curr_xmin);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::map<std::pair<int, int>, int> bbCandidateMatches; // map structure with pair of IDs of matched bounding boxes and counter for the ocurrences that pair happens

    for (auto match = matches.begin(); match != matches.end(); match++) // Iterate over the vector of matches
    {
        cv::KeyPoint prevKeyPoint, currKeyPoint;
        prevKeyPoint = prevFrame.keypoints[match->queryIdx]; // retrieve from the match pair the keypoint from the previous frame vector of keypoints
        currKeyPoint = currFrame.keypoints[match->trainIdx]; // retrieve from the match pair the keypoint from the current frame vector of keypoints
        
        BoundingBox prevBoundingBox, currBoundingBox;
        
        // find the Bounding Box from the previous frame vector of Bounding Boxes that contains the previous keypoint
        bool bb1Found = false; 
        for (auto boundingBox = prevFrame.boundingBoxes.begin(); boundingBox != prevFrame.boundingBoxes.end(); boundingBox++)
        {
            if (boundingBox->roi.contains(prevKeyPoint.pt))
            {
                if (!bb1Found)
                {
                    prevBoundingBox = *boundingBox;
                    bb1Found = true;
                }
                else // if the keypoint is found in more than one Bounding Box then disregard this keypoint/match since it can cross reference two unrelated bounding boxes
                {
                    bb1Found = false;
                    break;
                }       
            }
        }

        bool bb2Found = false;
        for (auto boundingBox = currFrame.boundingBoxes.begin(); boundingBox != currFrame.boundingBoxes.end(); boundingBox++)
        {
            if (boundingBox->roi.contains(currKeyPoint.pt))
            {
                if (!bb2Found)
                {
                    currBoundingBox = *boundingBox;
                    bb2Found = true;
                }
                else  // if the keypoint is found in more than one Bounding Box then disregard this keypoint/match since it can cross reference two unrelated bounding boxes
                {
                    bb2Found = false;
                    break;
                }
            }
        }

        if ( bb1Found && bb2Found ) //store the candidate pair of matched bounding boxes in a map that contains the candidate matches and the count of times this match is found.
        {
            std::pair<int,int> bbCandidateMatch = {prevBoundingBox.boxID,currBoundingBox.boxID};

            if (bbCandidateMatches.find(bbCandidateMatch) != bbCandidateMatches.end())
            {
                bbCandidateMatches[bbCandidateMatch]++;
            }
            else
            {
                bbCandidateMatches.insert(std::pair<std::pair<int,int>,int>(bbCandidateMatch,1));
            }
        }
    }

    for (auto it1 = bbCandidateMatches.begin(); it1 != bbCandidateMatches.end(); ++it1) // loop over the candidate matches and select, for each BB ID of the previous frame, the match with higher occurrences
    {
        int bb1ID = it1->first.first;
        int bb2ID = it1->first.second;
        int counter = it1->second;
        if (bbBestMatches.find(bb1ID) == bbBestMatches.end())
        {
            for (auto it2 = bbCandidateMatches.begin(); it2 != bbCandidateMatches.end(); ++it2)
            {
                if (bb1ID == it2->first.first)
                {
                    if (it2->second > counter)
                    {
                        bb2ID = it2->first.second;
                        counter = it2->second;
                    }
                }
            }
            bbBestMatches.insert(std::pair<int,int>(bb1ID,bb2ID));
        }
    }
    
}

void segmentPlane(std::vector<LidarPoint> &lidarPoints, int maxIterations, float distanceTol)
{
    std::unordered_set<int> inliersResult;
	srand(time(NULL));
	
	while (maxIterations--)
	{
		std::unordered_set<int> inliers;

		// Ensure getting 3 different random points
		while (inliers.size()<3)
			inliers.insert(rand()%(lidarPoints.size()));

		auto itr = inliers.begin();

        float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		//Eigen::Vector3f point1 = {lidarPoints[*itr].x,lidarPoints[*itr].y,lidarPoints[*itr].z};
		x1 = lidarPoints[*itr].x;
		y1 = lidarPoints[*itr].y;
		z1 = lidarPoints[*itr].z;
		itr++;
		//Eigen::Vector3f point2 = {lidarPoints[*itr].x,lidarPoints[*itr].y,lidarPoints[*itr].z};
		x2 = lidarPoints[*itr].x;
		y2 = lidarPoints[*itr].y;
		z2 = lidarPoints[*itr].z;
		itr++;
		//Eigen::Vector3f point3 = {lidarPoints[*itr].x,lidarPoints[*itr].y,lidarPoints[*itr].z};
		x3 = lidarPoints[*itr].x;
		y3 = lidarPoints[*itr].y;
		z3 = lidarPoints[*itr].z;
		//itr++;

		//Eigen::Vector3f vector12 = point2-point1;
		//Eigen::Vector3f vector13 = point3-point1;

		//Eigen::Vector3f vectorPlane = vector12.cross(vector13);

		float A = (y2-y1)*(z3-z1)-(z2-z1)*(y3-y1);
		float B = (z2-z1)*(x3-x1)-(x2-x1)*(z3-z1);
		float C = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
		
		// Ensure cross product is not null (when 3 random points happen to be in line and do not define a plane)
		//float norm =vectorPlane.norm();
		//if (norm == 0)
		if (A==0 && B==0 && C==0)
			continue;

		float D = -(A*x1+B*y1+C*z1);
		//float D = -vectorPlane.dot(point1);

		for(int index = 0; index < lidarPoints.size(); index++)
		{
			// Don't compute distance for the three originally selected random points
			if (inliers.count(index)>0)
				continue;

			float x,y,z; 
            x = lidarPoints[index].x;
            y = lidarPoints[index].y;
            z = lidarPoints[index].z;
			//Eigen::Vector3f point = {lidarPoints[index].x,lidarPoints[index].y,lidarPoints[index].z};
			float distance = fabs(A*x+B*y+C*z+D)/sqrt(A*A+B*B+C*C);
			//float distance = fabs(vectorPlane.dot(point)+D)/norm;
			if (distance < distanceTol)
			{
				inliers.insert(index);
			}
		}
		if (inliers.size()>inliersResult.size())
		{
			inliersResult=inliers;
		}
	}

    if (inliersResult.size() == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset" << std::endl;
    }

    int index = 0;
    for(auto it = lidarPoints.begin(); it != lidarPoints.end(); it++)
	{
		if(!inliersResult.count(index))
        {
            lidarPoints.erase(it);
            it--;
        }
        index++;
	}
}
