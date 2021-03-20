# Mid-Term Report - Writeup

## M.1 Data Buffer Optimization

Implementing ring-buffer functionality just requires adding two lines of code. By exceeding ```dataBuffer.size() > dataBufferSize``` the first object can easily be removed from the vector by using ```erase``` function.

```cpp
// push image into data frame buffer
DataFrame frame;
frame.cameraImg = img;
dataBuffer.push_back(frame);

if(dataBuffer.size() > dataBufferSize)
    dataBuffer.erase(dataBuffer.begin());
```

## MP.2 Keypoint Detection

As it is described bellow, Harris- and other keypoint detectors have been implemented in a very similar way to the code of the given ShiTomasi detector. Further a parameter ```timing``` has been added to the function to simplify performance measures.

```cpp
if (detectorType.compare("SHITOMASI") == 0)     detKeypointsShiTomasi(keypoints, imgGray, t1, false);
else if (detectorType.compare("HARRIS") == 0)   detKeypointsHarris(keypoints, imgGray, t1, false);
else                                            detKeypointsModern(keypoints, imgGray, detectorType, t1, false);
```

The Harris detector included the implementation of a ```non-maximum suppression``` functionality.
This can be achieved by iterating over the entire processed image. Pixel with a corner response larger ```minResponse``` will be selected as potential maximum keypoint. Further, intersection over union is caclulated with respect to a vector containing all other keypoints, which are already chosen to be maximum keypoints. Hereby, the keypoint size is understood as the region of a circle with the keypoint coordinates at the center. When the two keypoints should overlap, meaning having an IoU score larger than ```maxIoU```, the current keypoint is handeled as potential maximum keypoint. If the response value of the current keypoint should be larger than the one is already existing, the previous maximum point will be replaced by the current one.

```cpp
float maxIoU = 0.0;
// check for nms
for (int i=0;i<dst_norm_scaled.rows;i++)
    for (int j=0;j<dst_norm_scaled.cols;j++) {
        int response = dst_norm_scaled.at<uchar>(i,j);
        if (response >= minResponse) {
            // create new keypoint
            cv::KeyPoint kp;
            kp.pt.x = j; kp.pt.y = i;
            kp.size = 2*apertureSize;
            kp.response = response;

            // nms
            bool isMaxKeypoint = false;
            for(auto itr=keypoints.begin(); itr!=keypoints.end(); itr++) {
                float iou = cv::KeyPoint::overlap(kp, *itr);
                if(iou > maxIoU) {
                    isMaxKeypoint = true;
                    *itr = kp;
                    break;
                }
            }
            if (!isMaxKeypoint)
                keypoints.push_back(kp);
        }
    }
```

Other feature detectors didn't require any fruther implementation effort, where the OpenCV Library has mainly been used to provide the desired functionalities (FAST, BRISK, ORB, AKAZE, SIFT).

```cpp
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, float &timing, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    if      (detectorType.compare("FAST") == 0)  detector=cv::FastFeatureDetector::create();
    else if (detectorType.compare("BRISK") == 0) detector=cv::BRISK::create();
    else if (detectorType.compare("ORB") == 0)   detector=cv::ORB::create();
    else if (detectorType.compare("AKAZE") == 0) detector=cv::AKAZE::create();
    else if (detectorType.compare("SIFT") == 0)  detector=cv::xfeatures2d::SIFT::create();

    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints,cv::Mat());
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType + " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    timing =  1000 * t / 1.0;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
```


## MP.3 Keypoint Removal

OpenCV kindly provides methods, such as ```contains```, to check whether a point is located whithin a rectangular shape or not.

```cpp
cv::Rect vehicleRect(535, 180, 180, 150);
        bool bFocusOnVehicle = true;
        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> kp_temp;
            for(auto it=keypoints.begin(); it!=keypoints.end();it++)
                if(vehicleRect.contains(it->pt))
                    kp_temp.push_back(*it);
            keypoints=kp_temp;
        }
```

## MP.4 Keypoint Descriptors

Using the OpenCV library, no further coding effort is required. Descriptor methods are provided by several OpenCV Modules of ```feature2d``` (BRISK, ORB, AKAZE) and ```xfeatures2d``` (BRIEF, FREAK, SIFT, SURF).

```cpp
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, float &time)
{
    //// -> BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    time = 1000 * t / 1.0;
}
```

## MP.5 Descriptor Matching

Fast Library for Approximate Nearest Neighbours (FLANN) is used as alternative to Brute Force (BF) matcher. Rather than comparing every keypoint with each other from the current and the previous frame (brute force), FLANN provides an approximation of the nearest neighbour search by using a kd-tree datastructure. For a sorting and evaluation criteria both algorithms accept distance as input parameter.

```cpp
matcher = cv::FlannBasedMatcher::create();
```


## MP.6 KNN and Descriptor Distance Ratio

The brute force matcher would only rely on to the nearest neighbour. Hereby, a list of the corresponding descriptor distance values is sorted in descending order, which is the functions return from ```BFMatcher```. Compared to knn, this uses a selection of the k nearest neighbours. Furthermore, a threshold is introduced as ```minDescDistRatio``` to reduce the number of false positives while finding good keypoint pairs.

```cpp
if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)
    std::vector<std::vector<cv::DMatch>> matchesKNN;
    matcher->knnMatch(descSource, descRef, matchesKNN, 2);

    float minDescDistRatio = 0.8;
    for (auto it = matchesKNN.begin(); it != matchesKNN.end(); ++it)
        if ((*it)[0].distance < (minDescDistRatio * (*it)[1].distance))
            matches.push_back((*it)[0]);
}
```


## Performance Evaluation:

### M.7: Number of Detected Keypoints
| detector       | img_0 | img_1 | img_2 | img_3 | img_4 | img_5 | img_6 | img_7 | img_8 | img_9 | nbhd size |
|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------------:|
| SHITOMASI | 125   | 118   | 123   | 120   | 120   | 113   | 114   | 123   | 111   | 112   | 4                 |
| HARRIS    | 18    | 16    | 19    | 22    | 25    | 45    | 17    | 30    | 25    | 31    | 6                 |
| FAST      | **419**   | **427**   | **404**   | **423**   | **386**   | **414**   | **418**   | **406**   | **396**   | **401**   | 7                 |
| BRISK     | 264   | 282   | 282   | 277   | 297   | 279   | 289   | 272   | 266   | 254   | 21.9              |
| ORB       | 92    | 102   | 106   | 113   | 109   | 125   | 130   | 129   | 127   | 128   | 56                |
| AKAZE     | 166   | 157   | 161   | 155   | 163   | 164   | 173   | 175   | 177   | 179   | 7.7               |
| SIFT      | 138   | 132   | 124   | 137   | 134   | 140   | 137   | 148   | 159   | 137   | 5                 |



### M.8: Number of Matched Keypoints
* BF_MATCH
* SEL_KNN
* desc. dist. ratio = 0.8

| det/desc  | BRISK | BRIEF | ORB  | FREAK | AKAZE | SIFT |
|:---------:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|
| SHITOMASI | 767   | 944   | 907  | 768   | -     | 927  |
| HARRIS    | 112   | 143   | 139  | 104   | -     | 165  |
| FAST      | 2183  | **2831**  | **2762** | 2233  | -     | **2782** |
| BRISK     | 1570  | 1704  | 1510 | 1524  | -     | 1646 |
| ORB       | 751   | 545   | 761  | 420   | -     | 763  |
| AKAZE     | 1215  | 1266  | 1186 | 1187  | 1259  | 1270 |
| SIFT      | 592   | 702   | -    | 593   | -     | 800  |




### M.9: Perofrmance measure in [ms], detector/ descriptor

| det/desc  | BRISK | BRIEF | ORB   | FREAK  | AKAZE | SIFT   |
|:---------:|:-----:|:-----:|:-----:|:------:|:-----:|:------:|
| SHITOMASI | 14.61 | 15.15 | 18.02 | 43.64  | -     | 31.73  |
| HARRIS    | 15.65 | 17.19 | 19.72 | 45.04  | -     | 31.89  |
| FAST      | **5.77**  | **4**     | **6.13**  | 37.94  | -     | 25.91  |
| BRISK     | 35.52 | 35.81 | 46.06 | 66.43  | -     | 62.07  |
| ORB       | 8.12  | 8     | 19.38 | 39.67  | -     | 38.86  |
| AKAZE     | 56.14 | 54.21 | 62.16 | 84.08  | 95.51 | 72.05  |
| SIFT      | 89.03 | 97.62 | -     | 129.89 | -     | 147.06 |


Performance measured by number of matched keypoints:
1.	FAST/ BRIEF
2.	FAST/ SIFT
3.	FAST/ORB

Performance measured by execution time detector/descriptor in [ms]:
1.	FAST/ BRIEF
2.	FAST/ BRISK
3.	FAST/ ORB

The combination FAST/ BRIEF clearly stands out from other approaches in both runtime as well as number of detected keypoints pairs. Due to the very efficient and fast execution of binary detectors/descriptors the selected pair of algorithm should be an appropriate solution to also perform good within real-time applications. 	
Even though SIFT is in the position to achieve very good number of keypoint matches throughout all det/desc combinations, the gradient-based feature approach heavily suffers under long execution time, which makes the algorithm inappropriate to be applied within real-time application.
