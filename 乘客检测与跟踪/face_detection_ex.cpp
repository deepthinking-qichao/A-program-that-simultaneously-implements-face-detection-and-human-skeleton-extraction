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

//bool Distance(cv::Rect2d rect1,cv::Rect2d rect2);
double FaceConfidence(int match,int nomatch);
double Distance(cv::Rect2d rect1,cv::Rect2d rect2);
int matchFacePpList(const std::vector<cv::Rect2d> &faces,const FacePp &facePpinFrame);
int createId();
void createColor(std::vector<cv::Scalar> &color);

int passengerId = 0;

int main(int argc, char** argv)
{
	try
	{
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
		//std::string video = "F:/VS2013/video/425_clip(1).avi";		//factory video
		//std::string video = "F:/VS2013/video/ch04_20160227101430_clip.mp4";		//office right

		std::vector<cv::Scalar> trajectory_color;
		createColor(trajectory_color);

		std::string video = argv[1];		

		cv::VideoCapture cap(video);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;

		}

		//image_window win;

		// Load face detection model
		typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
		object_detector<image_scanner_type> detector;
        deserialize("face_detector.svm") >> detector;

        double rate = 25.0;  
    	cv::Size videoSize(roi_width,roi_height);  
    	cv::VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize); 

    	cv::MultiTracker *mul_tracker;
		mul_tracker = new cv::MultiTracker("KCF");
		
		int frameCount = -1;
		int skipFrame = 10;		//default is 10

		double faceBeg_T = 25;
		double faceEnd_T = -100;

		double confidence_up_T = 50;

		bool show_trajectory = false;

		double retention_T = 1;

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
			std::string temp = argv[3];
			if(temp == "true")
			{
				show_trajectory = true;		//default is false
			}
		}

		std::vector<cv::Scalar> color;
		cv::namedWindow("tracker");

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
			

			std::vector<cv::Rect2d> faces_rect;			//tracking rect
			std::cout<<"frame processing begin"<<std::endl;
			if (frameCount%skipFrame==0)
			{
				frameRoiClone = frameRoi.clone();
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

				std::vector<cv::Rect2d> faces_rect1;		//detection rect
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
			sPp = "people:" + sPp;
			cv::putText(frameRoi, sPp, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

			writer << frameRoi;
			cv::imshow("tracker", frameRoi);
			std::cout<<"frame processing end"<<std::endl;
		}
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

//distance between two rect
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

//caculate Confidence increment
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

//Find the nearest face rect from std::vector<cv::Rect2d> &faces
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
		//std::cout<<"\t"<<i<<": ";
		//std::cout<<"distance is "<<distance<<" distance_T is "<<distance_T<<std::endl;		//debug

		if (distance > distance_T) continue;
		if (distance<distance_min)
		{
			distance_min = distance;
			idx_min = i;
		}
	}
	return idx_min;
}

//creat color
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

//create Id
int createId()
{
	passengerId++;
	return passengerId;
}

