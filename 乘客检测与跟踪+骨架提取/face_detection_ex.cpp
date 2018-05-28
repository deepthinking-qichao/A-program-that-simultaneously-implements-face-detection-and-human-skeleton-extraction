#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h> 

#include <dlib/svm_threaded.h> 
#include <dlib/data_io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 

#include <opencv2/core/utility.hpp>  
#include <opencv2/tracking.hpp>  
#include <opencv2/videoio.hpp> 
#include <iostream>  
#include <cstring>
#include <sstream>  

#include <cmath>

#include <fstream>

// C++ std library dependencies
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread> // std::this_thread
// Other 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/headers.hpp>

// See all the available parameter options withe the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
DEFINE_bool(disable_multi_thread,       false,          "It would slightly reduce the frame rate in order to highly reduce the lag. Mainly useful"
                                                        " for 1) Cases where it is needed a low latency (e.g. webcam in real-time scenarios with"
                                                        " low-range GPU devices); and 2) Debugging OpenPose when it is crashing to locate the"
                                                        " error.");
DEFINE_int32(profile_speed,             1000,           "If PROFILER_ENABLED was set in CMake or Makefile.config files, OpenPose will show some"
                                                        " runtime statistics at this frame number.");
// Producer
DEFINE_string(image_dir,                "examples/media/",      "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// OpenPose
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
                                                        " machine.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y)"
                                                        " coordinates that will be saved with the `write_keypoint` & `write_keypoint_json` flags."
                                                        " Select `0` to scale it to the original source resolution, `1`to scale it to the net output"
                                                        " size (set with `net_resolution`), `2` to scale it to the final output size (set with"
                                                        " `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related"
                                                        " with `scale_number` and `scale_gap`.");
DEFINE_int32(number_people_max,         -1,             "This parameter will limit the maximum number of people detected, by keeping the people with"
                                                        " top scores. The score is based in person area over the image, body part score, as well as"
                                                        " joint score (between each pair of connected body parts). Useful if you know the exact"
                                                        " number of people in the scene, so it can remove false positives (if all the people have"
                                                        " been detected. However, it might also include false negatives by removing very small or"
                                                        " highly occluded people. -1 will keep them all.");
// OpenPose Body Pose
DEFINE_bool(body_disable,               false,          "Disable body keypoint detection. Option only possible for faster (but less accurate) face"
                                                        " keypoint detection.");
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
// OpenPose Body Pose Heatmaps and Part Candidates
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will fill op::Datum::poseHeatMaps array with the body part heatmaps, and"
                                                        " analogously face & hand heatmaps to op::Datum::faceHeatMaps & op::Datum::handHeatMaps."
                                                        " If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential"
                                                        " memory order: body parts + bkg + PAFs. It will follow the order on"
                                                        " POSE_BODY_PART_MAPPING in `src/openpose/pose/poseParameters.cpp`. Program speed will"
                                                        " considerably decrease. Not required for OpenPose, enable it only if you intend to"
                                                        " explicitly use this information later.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                                        " background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale,            2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [-1,1], 1 for [0,1]; 2 for integer"
                                                        " rounded [0,255]; and 3 for no scaling.");
DEFINE_bool(part_candidates,            false,          "Also enable `write_json` in order to save this information. If true, it will fill the"
                                                        " op::Datum::poseCandidates array with the body part candidates. Candidates refer to all"
                                                        " the detected body parts, before being assembled into people. Note that the number of"
                                                        " candidates is equal or higher than the number of final body parts (i.e. after being"
                                                        " assembled into people). The empty body parts are filled with 0s. Program speed will"
                                                        " slightly decrease. Not required for OpenPose, enable it only if you intend to explicitly"
                                                        " use this information.");
// OpenPose Face
DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`. Note that this will considerable slow down the performance and increse"
                                                        " the required GPU memory. In addition, the greater number of people on the image, the"
                                                        " slower OpenPose will be.");
DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the face keypoint"
                                                        " detector. 320x320 usually works fine while giving a substantial speed up when multiple"
                                                        " faces on the image.");
// OpenPose Hand
DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`. Analogously to `--face`, it will also slow down the performance, increase"
                                                        " the required GPU memory and its speed depends on the number of people.");
DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the hand keypoint"
                                                        " detector.");
DEFINE_int32(hand_scale_number,         1,              "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results"
                                                        " were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4.");
DEFINE_double(hand_scale_range,         0.4,            "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
                                                        " between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if"
                                                        " scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");
DEFINE_bool(hand_tracking,              false,          "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
                                                        " is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it"
                                                        " simply looks for hands in positions at which hands were located in previous frames, but"
                                                        " it does not guarantee the same person ID among frames.");
// OpenPose 3-D Reconstruction
DEFINE_bool(3d,                         false,          "Running OpenPose 3-D reconstruction demo: 1) Reading from a stereo camera system."
                                                        " 2) Performing 3-D reconstruction from the multiple views. 3) Displaying 3-D reconstruction"
                                                        " results. Note that it will only display 1 person. If multiple people is present, it will"
                                                        " fail.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF.");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
// OpenPose Rendering Pose
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_int32(render_pose,               -1,             "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g. `alpha_X` flags). If -1, it will pick CPU if"
                                                        " CPU_ONLY is enabled, or GPU if CUDA is enabled. If rendering is enabled, it will render"
                                                        " both `outputData` and `cvOutputData` with the original image and desired body part to be"
                                                        " shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_double(face_render_threshold,    0.4,            "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_int32(face_render,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(face_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(face_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_double(hand_render_threshold,    0.2,            "Analogous to `render_threshold`, but applied to the hand keypoints.");
DEFINE_int32(hand_render,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(hand_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(hand_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");
// Result Saving
DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV"
                                                        " function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
                                                        " final path does not finish in `.avi`. It internally uses cv::VideoWriter.");
DEFINE_string(write_json,               "",             "Directory to write OpenPose output in JSON format. It includes body, hand, and face pose"
                                                        " keypoints (2-D and 3-D), as well as pose candidates (if `--part_candidates` enabled).");
DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with JSON COCO validation format.");
DEFINE_string(write_heatmaps,           "",             "Directory to write body pose heatmaps in PNG format. At least 1 `add_heatmaps_X` flag"
                                                        " must be enabled.");
DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
                                                        " For lossless compression, recommended `png` for integer `heatmaps_scale` and `float` for"
                                                        " floating values.");
DEFINE_string(write_keypoint,           "",             "(Deprecated, use `write_json`) Directory to write the people pose keypoint data. Set format"
                                                        " with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format,    "yml",          "(Deprecated, use `write_json`) File extension and format for `write_keypoint`: json, xml,"
                                                        " yaml & yml. Json not available for OpenCV < 3.0, use `write_keypoint_json` instead.");
DEFINE_string(write_keypoint_json,      "",             "(Deprecated, use `write_json`) Directory to write people pose data in JSON format,"
                                                        " compatible with any OpenCV version."); 

using namespace dlib;
using namespace std;

struct FacePp
{
	std::vector<cv::Rect2d> faceList;
	cv::Rect2d face;
	double confidence;
	int faceMatch;
	int nofaceMatch;
	int idx;
};

struct PosePoint
{
	double x[18];
	double y[18];
	double c[18];
};

// If the user needs his own variables, he can inherit the op::Datum struct and add them
// UserDatum can be directly used by the OpenPose wrapper because it inherits from op::Datum, just define Wrapper<UserDatum> instead of
// Wrapper<op::Datum>
struct UserDatum : public op::Datum
{
    bool boolThatUserNeedsForSomeReason;

    UserDatum(const bool boolThatUserNeedsForSomeReason_ = false) :
        boolThatUserNeedsForSomeReason{boolThatUserNeedsForSomeReason_}
    {}
};

// The W-classes can be implemented either as a template or as simple classes given
// that the user usually knows which kind of data he will move between the queues,
// in this case we assume a std::shared_ptr of a std::vector of UserDatum

// This worker will just read and return all the jpg files in a directory
class UserInputClass
{
public:
    UserInputClass(const std::string& directoryPath) :
        mImageFiles{op::getFilesOnDirectory(directoryPath, "jpg")},
        // If we want "jpg" + "png" images
        // mImageFiles{op::getFilesOnDirectory(directoryPath, std::vector<std::string>{"jpg", "png"})},
        mCounter{0},
        mClosed{false}
    {
        if (mImageFiles.empty())
            op::error("No images found on: " + directoryPath, __LINE__, __FUNCTION__, __FILE__);
    }

    std::shared_ptr<std::vector<UserDatum>> createDatum()
    {
        // Close program when empty frame
        if (mClosed || mImageFiles.size() <= mCounter)
        {
            op::log("Last frame read and added to queue. Closing program after it is processed.", op::Priority::High);
            // This funtion stops this worker, which will eventually stop the whole thread system once all the frames
            // have been processed
            mClosed = true;
            return nullptr;
        }
        else // if (!mClosed)
        {
            // Create new datum
            auto datumsPtr = std::make_shared<std::vector<UserDatum>>();
            datumsPtr->emplace_back();
            auto& datum = datumsPtr->at(0);

            // Fill datum
            datum.cvInputData = cv::imread(mImageFiles.at(mCounter++));

            // If empty frame -> return nullptr
            if (datum.cvInputData.empty())
            {
                op::log("Empty frame detected on path: " + mImageFiles.at(mCounter-1) + ". Closing program.",
                        op::Priority::High);
                mClosed = true;
                datumsPtr = nullptr;
            }

            return datumsPtr;
        }
    }

    bool isFinished() const
    {
        return mClosed;
    }

private:
    const std::vector<std::string> mImageFiles;
    unsigned long long mCounter;
    bool mClosed;
};

// This worker will just read and return all the jpg files in a directory
class UserOutputClass
{
public:
    bool display(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr,cv::Mat& poseFrame)
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        char key = ' ';
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
            poseFrame = datumsPtr->at(0).cvOutputData;
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            key = (char)cv::waitKey(1);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        return (key == 27);
    }
    void printKeypoints(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr,std::vector<PosePoint>& posePoints)
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            op::log("\nKeypoints:");
            // Accesing each element of the keypoints
            const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
            op::log("Person pose keypoints:");
            for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
            {
                op::log("Person " + std::to_string(person) + " (x, y, score):");
                PosePoint tmpPosePoint;
                for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
                {
                    std::string valueToPrint;               
                    for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)                   
                        valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
                    tmpPosePoint.x[bodyPart]=poseKeypoints[{person, bodyPart, 0}];
                    tmpPosePoint.y[bodyPart]=poseKeypoints[{person, bodyPart, 1}];
                    tmpPosePoint.c[bodyPart]=poseKeypoints[{person, bodyPart, 2}];
                    op::log(valueToPrint);
                }
                posePoints.push_back(tmpPosePoint);
            }
            op::log(" ");
            /*
            // Alternative: just getting std::string equivalent
            op::log("Face keypoints: " + datumsPtr->at(0).faceKeypoints.toString());
            op::log("Left hand keypoints: " + datumsPtr->at(0).handKeypoints[0].toString());
            op::log("Right hand keypoints: " + datumsPtr->at(0).handKeypoints[1].toString());
            // Heatmaps
            const auto& poseHeatMaps = datumsPtr->at(0).poseHeatMaps;
            if (!poseHeatMaps.empty())
            {
                op::log("Pose heatmaps size: [" + std::to_string(poseHeatMaps.getSize(0)) + ", "
                        + std::to_string(poseHeatMaps.getSize(1)) + ", "
                        + std::to_string(poseHeatMaps.getSize(2)) + "]");
                const auto& faceHeatMaps = datumsPtr->at(0).faceHeatMaps;
                op::log("Face heatmaps size: [" + std::to_string(faceHeatMaps.getSize(0)) + ", "
                        + std::to_string(faceHeatMaps.getSize(1)) + ", "
                        + std::to_string(faceHeatMaps.getSize(2)) + ", "
                        + std::to_string(faceHeatMaps.getSize(3)) + "]");
                const auto& handHeatMaps = datumsPtr->at(0).handHeatMaps;
                op::log("Left hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", "
                        + std::to_string(handHeatMaps[0].getSize(1)) + ", "
                        + std::to_string(handHeatMaps[0].getSize(2)) + ", "
                        + std::to_string(handHeatMaps[0].getSize(3)) + "]");
                op::log("Right hand heatmaps size: [" + std::to_string(handHeatMaps[1].getSize(0)) + ", "
                        + std::to_string(handHeatMaps[1].getSize(1)) + ", "
                        + std::to_string(handHeatMaps[1].getSize(2)) + ", "
                        + std::to_string(handHeatMaps[1].getSize(3)) + "]");
            }
            */
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
    }
};

//bool Distance(cv::Rect2d rect1,cv::Rect2d rect2);
double FaceConfidence(int match,int nomatch);
double Distance(cv::Rect2d rect1,cv::Rect2d rect2);
int matchFacePpList(const std::vector<cv::Rect2d> &faces,const FacePp &facePpinFrame);
int createId();
void createColor(std::vector<cv::Scalar> &color);

int passengerId = 0;

int openPoseTutorialWrapper3(std::vector<PosePoint>& posePoints,cv::Mat& poseFrame)
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
              __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::Profiler::setDefaultX(FLAGS_profile_speed);

    op::log("Starting pose estimation demo.", op::Priority::High);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration - Google flags to program variables
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
    // faceNetInputSize
    const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
    // handNetInputSize
    const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // JSON saving
    const auto writeJson = (!FLAGS_write_json.empty() ? FLAGS_write_json : FLAGS_write_keypoint_json);
    if (!FLAGS_write_keypoint.empty() || !FLAGS_write_keypoint_json.empty())
        op::log("Flags `write_keypoint` and `write_keypoint_json` are deprecated and will eventually be removed."
                " Please, use `write_json` instead.", op::Priority::Max);
    // keypointScale
    const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                  FLAGS_heatmaps_add_PAFs);
    const auto heatMapScale = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
    // Enabling Google Logging
    const bool enableGoogleLogging = true;
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // Configure OpenPose
    op::Wrapper<std::vector<UserDatum>> opWrapper{op::ThreadManagerMode::Asynchronous};
    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    const op::WrapperStructPose wrapperStructPose{!FLAGS_body_disable, netInputSize, outputSize, keypointScale,
                                                  FLAGS_num_gpu, FLAGS_num_gpu_start, FLAGS_scale_number,
                                                  (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, FLAGS_3d),
                                                  poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose,
                                                  (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, FLAGS_model_folder,
                                                  heatMapTypes, heatMapScale, FLAGS_part_candidates,
                                                  (float)FLAGS_render_threshold, FLAGS_number_people_max,
                                                  enableGoogleLogging, FLAGS_3d};
    /*                                        
    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{FLAGS_face, faceNetInputSize,
                                                  op::flagsToRenderMode(FLAGS_face_render, FLAGS_3d, FLAGS_render_pose),
                                                  (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap,
                                                  (float)FLAGS_face_render_threshold};
    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number,
                                                  (float)FLAGS_hand_scale_range, FLAGS_hand_tracking,
                                                  op::flagsToRenderMode(FLAGS_hand_render, FLAGS_3d, FLAGS_render_pose),
                                                  (float)FLAGS_hand_alpha_pose, (float)FLAGS_hand_alpha_heatmap,
                                                  (float)FLAGS_hand_render_threshold};
    */
    // Consumer (comment or use default argument to disable any output)
    const auto displayMode = op::DisplayMode::NoDisplay;
    const bool guiVerbose = false;
    const bool fullScreen = false;
    const op::WrapperStructOutput wrapperStructOutput{displayMode, guiVerbose, fullScreen, FLAGS_write_keypoint,
                                                      op::stringToDataFormat(FLAGS_write_keypoint_format),
                                                      writeJson, FLAGS_write_coco_json,
                                                      FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
                                                      FLAGS_write_heatmaps, FLAGS_write_heatmaps_format};
    // Configure wrapper
    op::log("Configuring OpenPose wrapper.", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    /*
    opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, op::WrapperStructInput{},
                        wrapperStructOutput);*/
    opWrapper.configure(wrapperStructPose, op::WrapperStructInput{},
                        wrapperStructOutput);
    // Set to single-thread running (to debug and/or reduce latency)
    if (FLAGS_disable_multi_thread)
       opWrapper.disableMultiThreading();

    op::log("Starting thread(s)", op::Priority::High);
    opWrapper.start();

    // User processing
    UserInputClass userInputClass(FLAGS_image_dir);
    UserOutputClass userOutputClass;
    bool userWantsToExit = false;
    while (!userWantsToExit && !userInputClass.isFinished())
    {
        // Push frame
        auto datumToProcess = userInputClass.createDatum();
        if (datumToProcess != nullptr)
        {
            auto successfullyEmplaced = opWrapper.waitAndEmplace(datumToProcess);
            // Pop frame
            std::shared_ptr<std::vector<UserDatum>> datumProcessed;
            if (successfullyEmplaced && opWrapper.waitAndPop(datumProcessed))
            {
                userWantsToExit = userOutputClass.display(datumProcessed,poseFrame);
                userOutputClass.printKeypoints(datumProcessed,posePoints);
            }
            else
                op::log("Processed datum could not be emplaced.", op::Priority::High,
                        __LINE__, __FUNCTION__, __FILE__);
        }
    }

    op::log("Stopping thread(s)", op::Priority::High);
    opWrapper.stop();

    // Measuring total time
    const auto now = std::chrono::high_resolution_clock::now();
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
                            * 1e-9;
    const auto message = "Real-time pose estimation demo successfully finished. Total time: "
                       + std::to_string(totalTimeSec) + " seconds.";
    op::log(message, op::Priority::High);

    return 0;
}

int main(int argc, char** argv)
{
	try
	{
		// Parsing command line flags
    	gflags::ParseCommandLineFlags(&argc, &argv, true);

		std::cout << "Hello, Let's begin a test of cout to file." << std::endl;
		std::streambuf* coutBuf = std::cout.rdbuf();
		std::ofstream of("out.txt");
		std::streambuf* fileBuf = of.rdbuf();
		std::cout.rdbuf(fileBuf);

		//set roi
		int roi_tl_x =0,roi_tl_y=0,roi_width=749,roi_height=720;		//roi in office left
		//int roi_tl_x =0,roi_tl_y=0,roi_width=651,roi_height=720;		//roi in subway left
		//cv::Rect roi(817, 0, 1144, 1440);	//roi in factory
		//cv::Rect roi(1083, 0, 1455, 1440);	//roi in office right
		cv::Rect roi(roi_tl_x*2, roi_tl_y*2, roi_width*2, roi_height*2);	//roi in office left
		cv::Rect roi_half(roi_tl_x, roi_tl_y, roi_width, roi_height);	//roi in office left
		//cv::Rect roi(829, 34, 470, 666);	//roi in office left cut

		std::vector<cv::Scalar> trajectory_color;
		createColor(trajectory_color);

		std::string video = argv[1];		

		cv::VideoCapture cap(video);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;

		}

		// Load face detection and pose estimation models.
		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
		object_detector<image_scanner_type> detector;
        deserialize("face_detector.svm") >> detector;

        double rate = 25.0;  
    	cv::Size videoSize(roi_width,roi_height);  
    	cv::VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize); 
    	cv::VideoWriter writer_pose("VideoTest_pose.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize); 

    	cv::MultiTracker *mul_tracker;
		mul_tracker = new cv::MultiTracker("KCF");
		int frameCount = -1;
		int skipFrame = 10;		//default is 10

		double faceBeg_T = 25;
		double faceEnd_T = -100;

		double confidence_up_T = 50;

		bool show_trajectory = false;

		double retention_T = 1;

		double T_escalatorEntryLine_y = 30;

		std::vector<FacePp> facePpinFrame;
		facePpinFrame.clear();

		if (argc >= 3)
		{
			std::stringstream ssGap;
			int sGap;
			ssGap << argv[2];
			ssGap >> sGap;
			skipFrame = sGap;		//default is 10
		}

		if (argc >= 4)
		{
			std::stringstream ssTrajectory;
			int sTrajectory;
			ssTrajectory << argv[3];
			ssTrajectory >> sTrajectory;
			show_trajectory = sTrajectory;		//default is true
		}

		std::vector<cv::Scalar> color;
		cv::namedWindow("tracker");
		cv::namedWindow("User worker GUI");

		cv::FileStorage fs("posepoint.xml", cv::FileStorage::WRITE);
		std::vector<std::vector<PosePoint> >framePosePoints;
		fs << "frame" << "[";

		if (!fs.isOpened())  
		{  
			std::cerr << "failed to open xml" << std::endl;  
		}  

		// Grab and process frames until the main window is closed by the user.
		while (1)
		{
			double fps = (double)cv::getTickCount();

			if (cv::waitKey(1) == ' ')
				cv::waitKey();

			// Grab a frame
			cv::Mat temp,frame,frameRoi,frameRoiClone;
			cap >> frame;
			frameCount++;
			if (!frame.data || cv::waitKey(1) == 27)        
			{
				std::cout<<"frame is unaviliable or press Esc, the video showing go to end"<<std::endl;
				break;
			}
			frameRoi = frame(roi_half);
			
			// Running openPoseTutorialWrapper3
    		cv::imwrite("examples/media/frame.jpg",frameRoi);
    		std::vector<PosePoint> posePoints;
    		cv::Mat poseFrame;
    		openPoseTutorialWrapper3(posePoints,poseFrame);
    		writer_pose << poseFrame;

			std::vector<cv::Rect2d> faces_rect;			//tracking rect
			if (frameCount%skipFrame==0)
			{
				frameRoiClone =frameRoi.clone();
				temp = frame.clone();
				cv::pyrUp(temp, temp);
				// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
				// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
				// long as temp is valid.  Also don't do anything to temp that would cause it
				// to reallocate the memory which stores the image as that will make cimg
				// contain dangling pointers.  This basically means you shouldn't modify temp
				// while using cimg.
				cv_image<bgr_pixel> cimg(temp(roi));
				// Detect faces 
				std::vector<rectangle> faces = detector(cimg);

				// Display it all on the screen
				/*
				win.clear_overlay();
				win.set_image(cimg);
				win.add_overlay(faces, rgb_pixel(255, 0, 0));
				*/

				std::vector<cv::Rect2d> faces_rect1;		//detection rect
				//cv::Mat img = dlib::toMat(faces);
				for (unsigned long j = 0; j < faces.size(); ++j)
				{
					cv::Rect2d faceTmp;
					faceTmp.x = faces[j].left() / 2 ;
					faceTmp.y = faces[j].top() / 2;
					faceTmp.width = faces[j].width() / 2;
					faceTmp.height = faces[j].height() / 2;
					faces_rect1.push_back(faceTmp);
				}

				//track function
				faces_rect.clear();
				mul_tracker->update(frameRoi, faces_rect);
				if(faces_rect.size()==0)
					std::cout<<"people tracking update is zero"<<std::endl;

				//update the tracking face in the current frame using KCF
				for (std::vector<FacePp>::iterator it = facePpinFrame.begin(); it != facePpinFrame.end();it++)
				{
					int match_idx = matchFacePpList(faces_rect,(*it));
					if (match_idx >=0)
						(*it).face = faces_rect[match_idx];
				}

				//debug statement
				std::cout<<"The "<<frameCount<<" frame(detection):"<<std::endl;
				int m = facePpinFrame.size();
				int n = faces_rect1.size();
				std::cout<<"tracking num is "<<m<<"  detection num is "<<n<<std::endl;
				int *detectFlag = new int[n];
				for(int k=0;k<n;k++)
					detectFlag[k]=0;

				for (int i=0;i<m;i++)			//facePpinFrame rect
				{
					std::cout<<"\t"<<facePpinFrame[i].idx<<" people face: "<<std::endl;
					int matchIdx = matchFacePpList(faces_rect1,facePpinFrame[i]);
					if(matchIdx>=0)		//detection to facePpList update
					{
						//trackFlag[i]++;
						detectFlag[matchIdx]++;
						facePpinFrame[i].face = faces_rect1[matchIdx];
						facePpinFrame[i].faceList.push_back(facePpinFrame[i].face);
						facePpinFrame[i].faceMatch++;
						facePpinFrame[i].nofaceMatch = 0;
						facePpinFrame[i].confidence += FaceConfidence(facePpinFrame[i].faceMatch,facePpinFrame[i].nofaceMatch);
						if(facePpinFrame[i].confidence>confidence_up_T)
							facePpinFrame[i].confidence = confidence_up_T;
					}
					else		//add tracking target
					{
						facePpinFrame[i].faceList.push_back(facePpinFrame[i].face);
						facePpinFrame[i].faceMatch = 0;
						//facePpinFrame[i].nofaceMatch++;		//thinking about it 
						facePpinFrame[i].confidence += FaceConfidence(facePpinFrame[i].faceMatch,facePpinFrame[i].nofaceMatch);
					}
				}

				//add detection target
				for(int j=0;j<n;j++)
				{
					if(detectFlag[j]==0)
					{
						FacePp facePp_tmp;
						facePp_tmp.idx = createId();
						facePp_tmp.face=faces_rect1[j];
						facePp_tmp.faceList.push_back(facePp_tmp.face);
						facePp_tmp.faceMatch = 1;
						facePp_tmp.nofaceMatch = 0;
						facePp_tmp.confidence = faceBeg_T + FaceConfidence(facePp_tmp.faceMatch,facePp_tmp.nofaceMatch);
						facePpinFrame.push_back(facePp_tmp);
					}
				}

				//compare faceEnd_T to delete or retain facePp in facePpinFrame
				std::vector<FacePp>::iterator it = facePpinFrame.begin();
				for (;it != facePpinFrame.end();)
		        {
		            if ((*it).confidence < faceEnd_T)
		            {
		                it = facePpinFrame.erase(it);
		            }
		            else
		            {		            	
		                it++;
		            }
		        }

		        //if two or more element in facePpinFrame is same,delete one of them
		        bool *overlapFlag = new bool[facePpinFrame.size()];
				for(int k=0;k<facePpinFrame.size();k++)
					overlapFlag[k] = false;
				for(int j=0;j<n;j++)
		        {
		        	if(detectFlag[j]<2) continue;
		        	double distance_min = 10000;
		        	int idx_min = -1;
		        	for(int i=0;i<facePpinFrame.size();i++)
		        	{
		        		if(faces_rect1[j]==facePpinFrame[i].face)
		        		{
		        			overlapFlag[i]=true;
		        			int faceList_size = facePpinFrame[i].faceList.size();
		        			if(faceList_size-2 >= 0)
		        			{
		        				double distance = Distance(facePpinFrame[i].faceList[faceList_size-2],faces_rect1[j]);
		        				if (distance<distance_min)
			        			{
			        				distance_min=distance;
			        				idx_min = i;
			        			}	
		        			}
		        			else
		        				idx_min = i;	        			
		        		}
		        	}
		        	if(idx_min>=0)
		        		overlapFlag[idx_min]=false;
		        }

		        int overlap_count=0;
		        for(std::vector<FacePp>::iterator it1 = facePpinFrame.begin();it1!=facePpinFrame.end();)
	        	{
	        		if(overlapFlag[overlap_count]==true)
	        		{
	        			it1=facePpinFrame.erase(it1);		        				        			
	        		}
	        		else
	        		{
	        			it1++;		
	        		}
	        		overlap_count++;
	        	}

	        	delete [] overlapFlag;
	        	delete [] detectFlag;

		        //compare faceBeg_T to show facePp in window
		        std::vector<cv::Rect2d> facesExist;
		        int showFaceNum = 0;
		        for (std::vector<FacePp>::iterator it = facePpinFrame.begin(); it != facePpinFrame.end();it++)
		        {
		        	facesExist.push_back((*it).face);
		            if ((*it).confidence >= faceBeg_T)
		            {
		            	int color_idx = (*it).idx % trajectory_color.size();
		                cv::rectangle(frameRoi, (*it).face, trajectory_color[color_idx], 2, 1);			//detection frame show
		                cv::Point Pos((*it).face.x, (*it).face.y);
		                std::stringstream ssIdx;
						std::string sIdx;
						ssIdx << (*it).idx;
						ssIdx >> sIdx;
						std::stringstream ssConf;
						std::string sConf;
						ssConf << (*it).confidence;
						ssConf >> sConf;
		                std::string text = "Idx:" + sIdx + " Conf:" + sConf;
		                cv::putText(frameRoi, text, Pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
						showFaceNum++;
						std::cout<<"\t"<<text<<std::endl;
						if(show_trajectory)
						{
							for(std::vector<cv::Rect2d>::iterator itTrajectory = (*it).faceList.begin(); itTrajectory != (*it).faceList.end();itTrajectory++)
							{
								cv::Point2d tmpCenter;
								tmpCenter.x= (*itTrajectory).x+(*itTrajectory).width/2;
								tmpCenter.y= (*itTrajectory).y+(*itTrajectory).height/2;
								circle(frameRoi, tmpCenter, 2, trajectory_color[color_idx]);
							}							
						}
		            }
		        }

				std::cout<<"people face exist num is "<<facePpinFrame.size();
				std::cout<<"  people face show num is "<<showFaceNum<<std::endl;
				
				delete mul_tracker;
				mul_tracker = NULL;
				mul_tracker = new cv::MultiTracker("KCF");
				mul_tracker->add(frameRoiClone, facesExist);				
			}
			else
			{
				std::cout<<"The "<<frameCount<<" frame(tracking):"<<std::endl;
				faces_rect.clear();
				mul_tracker->update(frameRoi, faces_rect);

				for (std::vector<FacePp>::iterator it = facePpinFrame.begin(); it != facePpinFrame.end();it++)
				{
					int match_idx = matchFacePpList(faces_rect,(*it));
					if (match_idx >=0)
						(*it).face = faces_rect[match_idx];
					(*it).faceList.push_back((*it).face);
					if(((*it).faceList.size()-2>=0) && (Distance((*it).faceList[(*it).faceList.size()-2],(*it).face)<retention_T) && ((*it).faceMatch==0))					
						(*it).nofaceMatch++;	
				}

		        for (std::vector<FacePp>::iterator it = facePpinFrame.begin(); it != facePpinFrame.end();it++)
		        {
		            if ((*it).confidence >= faceBeg_T)
		            {
		                int color_idx = (*it).idx % trajectory_color.size();
		                cv::rectangle(frameRoi, (*it).face, trajectory_color[color_idx], 2, 1);			//detection frame show
                        cv::Point Pos((*it).face.x, (*it).face.y);
                        std::stringstream ssIdx;
        				std::string sIdx;
        				ssIdx << (*it).idx;
        				ssIdx >> sIdx;
        				std::stringstream ssConf;
        				std::string sConf;
        				ssConf << (*it).confidence;
        				ssConf >> sConf;
                        std::string text = "Idx:" + sIdx + " Conf:" + sConf;
                        cv::putText(frameRoi, text, Pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
                        std::cout<<"\t"<<text<<std::endl;
                        if(show_trajectory)
						{
							for(std::vector<cv::Rect2d>::iterator itTrajectory = (*it).faceList.begin(); itTrajectory != (*it).faceList.end();itTrajectory++)
							{
								cv::Point2d tmpCenter;
								tmpCenter.x= (*itTrajectory).x+(*itTrajectory).width/2;
								tmpCenter.y= (*itTrajectory).y+(*itTrajectory).height/2;
								circle(frameRoi, tmpCenter, 2, trajectory_color[color_idx]);
							}							
						}
		            }
		        }
			}

    		fps = (double)cv::getTickFrequency() / ((double)cv::getTickCount() - fps);
			//cout << "fps:" << fps << endl;
			cv::Point disPos(5, 20);
			std::stringstream ssFps;
			std::string sFps;
			ssFps << fps;
			ssFps >> sFps;
			sFps = "fps:" + sFps;
			cv::putText(frameRoi, sFps, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

			disPos.y += 20;
			std::stringstream ssPp;
			std::string sPp;
			ssPp << faces_rect.size();
			ssPp >> sPp;
			sPp = "peopleFace:" + sPp;
			cv::putText(frameRoi, sPp, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

			for (std::vector<PosePoint>::iterator it = posePoints.begin(); it != posePoints.end();)
	        {
	        	std::cout<<"people nose y axis is "<<(*it).y[0]<<std::endl;
	        	if((*it).y[0]<T_escalatorEntryLine_y)
	        	{
	        		std::cout<<"abandoned people nose y axis is "<<(*it).y[0]<<std::endl;
	        		it=posePoints.erase(it);
	        	}
	        	else
	        		it++;
	        }

	        disPos.y += 20;
			std::stringstream ssPp1;
			std::string sPp1;
			ssPp1 << posePoints.size();
			ssPp1 >> sPp1;
			sPp1 = "people:" + sPp1;
			cv::putText(frameRoi, sPp1, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

			disPos.y += 20;
			std::stringstream ssFrameCount;
			std::string sFrameCount;
			ssFrameCount << frameCount;
			ssFrameCount >> sFrameCount;
			sFrameCount = "frame:" + sFrameCount;
			cv::putText(frameRoi, sFrameCount, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

            //Skeleton extraction
			for (std::vector<PosePoint>::iterator it = posePoints.begin(); it != posePoints.end();it++)
			{
				for(int j=0;j<18;j++)
				{
					cv::Point2d tmpCenter;
					tmpCenter.x= (*it).x[j];
					tmpCenter.y= (*it).y[j];
					if((*it).c[j]!=0)
						cv::circle(frameRoi, tmpCenter, (*it).c[j]*5, cv::Scalar(255, 0, 0),-1);
				}
				cv::Point2d tmpCenter_start,tmpCenter_end;
				int lineThickness = 3;			
				tmpCenter_start.x= (*it).x[0];
				tmpCenter_start.y= (*it).y[0];
				tmpCenter_end.x= (*it).x[1];
				tmpCenter_end.y= (*it).y[1];
				if((*it).c[0]!=0 &&(*it).c[1]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[1];
				tmpCenter_start.y= (*it).y[1];
				tmpCenter_end.x= (*it).x[2];
				tmpCenter_end.y= (*it).y[2];
				if((*it).c[1]!=0 &&(*it).c[2]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[1];
				tmpCenter_start.y= (*it).y[1];
				tmpCenter_end.x= (*it).x[5];
				tmpCenter_end.y= (*it).y[5];
				if((*it).c[1]!=0 &&(*it).c[5]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[1];
				tmpCenter_start.y= (*it).y[1];
				tmpCenter_end.x= (*it).x[8];
				tmpCenter_end.y= (*it).y[8];
				if((*it).c[1]!=0 &&(*it).c[8]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[1];
				tmpCenter_start.y= (*it).y[1];
				tmpCenter_end.x= (*it).x[11];
				tmpCenter_end.y= (*it).y[11];
				if((*it).c[1]!=0 &&(*it).c[11]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[2];
				tmpCenter_start.y= (*it).y[2];
				tmpCenter_end.x= (*it).x[3];
				tmpCenter_end.y= (*it).y[3];
				if((*it).c[2]!=0 &&(*it).c[3]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[3];
				tmpCenter_start.y= (*it).y[3];
				tmpCenter_end.x= (*it).x[4];
				tmpCenter_end.y= (*it).y[4];
				if((*it).c[3]!=0 &&(*it).c[4]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[5];
				tmpCenter_start.y= (*it).y[5];
				tmpCenter_end.x= (*it).x[6];
				tmpCenter_end.y= (*it).y[6];
				if((*it).c[5]!=0 &&(*it).c[6]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[6];
				tmpCenter_start.y= (*it).y[6];
				tmpCenter_end.x= (*it).x[7];
				tmpCenter_end.y= (*it).y[7];
				if((*it).c[6]!=0 &&(*it).c[7]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[8];
				tmpCenter_start.y= (*it).y[8];
				tmpCenter_end.x= (*it).x[9];
				tmpCenter_end.y= (*it).y[9];
				if((*it).c[8]!=0 &&(*it).c[9]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[9];
				tmpCenter_start.y= (*it).y[9];
				tmpCenter_end.x= (*it).x[10];
				tmpCenter_end.y= (*it).y[10];
				if((*it).c[9]!=0 &&(*it).c[10]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[11];
				tmpCenter_start.y= (*it).y[11];
				tmpCenter_end.x= (*it).x[12];
				tmpCenter_end.y= (*it).y[12];
				if((*it).c[11]!=0 &&(*it).c[12]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				tmpCenter_start.x= (*it).x[12];
				tmpCenter_start.y= (*it).y[12];
				tmpCenter_end.x= (*it).x[13];
				tmpCenter_end.y= (*it).y[13];
				if((*it).c[12]!=0 &&(*it).c[13]!=0)
					cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);				
			}

			framePosePoints.push_back(posePoints);
            fs << "{";      //frame
            fs << "frameNum" << frameCount;
			fs << "people" << "[";
			for (std::vector<PosePoint>::iterator it = posePoints.begin(); it != posePoints.end();it++)
			{
                fs << "{";      //people
				fs << "joint" << "[";
				for(int j=0;j<18;j++)
				{
					fs << "{";
					fs << "x_value" << (*it).x[j];
					fs << "y_value" << (*it).y[j];
					fs << "confidence" << (*it).c[j];
					fs << "}";
				}
				fs << "]";
                fs << "}";      //people
			}
			fs << "]";
            fs << "}";      //frame

			writer << frameRoi;
			cv::imshow("tracker", frameRoi);
		}
		fs << "]";
		fs.release(); 
		of.flush();
		of.close();
		std::cout.rdbuf(coutBuf);
		std::cout << "Write Personal Information over..." << std::endl;
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

double Distance(cv::Rect2d rect1,cv::Rect2d rect2)
{
	cv::Point2d rect1_c,rect2_c;
	rect1_c.x = rect1.x + rect1.width / 2;
	rect1_c.y = rect1.y + rect1.height / 2;
	rect2_c.x = rect2.x + rect2.width / 2;
	rect2_c.y = rect2.y + rect2.height / 2;

	double distance = sqrt(pow(rect1_c.x-rect2_c.x,2)+pow(rect1_c.y-rect2_c.y,2));
	return distance;
}

double FaceConfidence(int match,int nomatch)
{
	double faceCof1 = 1;
	double faceCof2 = 1;
	int match_pre = match -1;
	int nomatch_pre = nomatch -1;
	if(match_pre<0) match_pre=0;
	if(nomatch_pre<0) nomatch_pre=0;

	double confidence = faceCof1 * pow(match,2) - faceCof1 * pow(match_pre,2) -(faceCof2 * pow(nomatch,3)-faceCof2 * pow(nomatch_pre,3));
	return confidence;
}

int matchFacePpList(const std::vector<cv::Rect2d> &faces,const FacePp &facePpinFrame)
{
	double distance_min = 10000;
	double distance;
	cv::Point2d rect1_c,rect2_c;
	int idx_min = -1;

	double dis_scale = 0.4;		//default is 0.7
	double distance_T;

	rect2_c.x = facePpinFrame.face.x + facePpinFrame.face.width / 2;
	rect2_c.y = facePpinFrame.face.y + facePpinFrame.face.height / 2;

	for(int i=0;i<faces.size();i++)
	{
		distance_T = (faces[i].width + facePpinFrame.face.width)/2 * dis_scale;
		rect1_c.x = faces[i].x + faces[i].width / 2;
		rect1_c.y = faces[i].y + faces[i].height / 2;	
		distance = sqrt(pow(rect1_c.x-rect2_c.x,2)+pow(rect1_c.y-rect2_c.y,2));
		std::cout<<"\t"<<i<<": ";
		std::cout<<"distance is "<<distance<<" distance_T is "<<distance_T<<std::endl;		//debug

		if (distance > distance_T) continue;
		if (distance<distance_min)
		{
			distance_min = distance;
			idx_min = i;
		}
	}
	return idx_min;
}

void createColor(std::vector<cv::Scalar> &color)
{
	color.push_back(cv::Scalar(255, 0, 0));
	color.push_back(cv::Scalar(0, 255, 0));
	color.push_back(cv::Scalar(0, 0, 255));
	color.push_back(cv::Scalar(255, 255, 0));
	color.push_back(cv::Scalar(0, 255, 255));
	color.push_back(cv::Scalar(255, 0, 255));
	color.push_back(cv::Scalar(255, 128, 0));
	color.push_back(cv::Scalar(255, 0, 128));
	color.push_back(cv::Scalar(0, 255, 128));
	color.push_back(cv::Scalar(128, 255, 0));
}

int createId()
{
	passengerId++;
	return passengerId;
}


