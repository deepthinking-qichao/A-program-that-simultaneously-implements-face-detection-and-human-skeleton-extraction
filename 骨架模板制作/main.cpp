#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

using namespace std;

struct PosePoint
{
	double x[18];
	double y[18];
	double c[18];
};

struct PoseCos
{
	double alpha[13];
	double beta[13];
};

void computeCos(double x1,double x2,double y1,double y2,bool confidenceFlag,PoseCos &poseCos,int idx);

int main(int argc, char** argv)
{
	std::streambuf* coutBuf = std::cout.rdbuf();
	std::ofstream of("out.txt");
	std::streambuf* fileBuf = of.rdbuf();
	std::cout.rdbuf(fileBuf);

	/*
	//abnormal template 
	std::vector<int> templateNum;
	templateNum.push_back(0);
	templateNum.push_back(596);
	templateNum.push_back(658);
	templateNum.push_back(1013);
	templateNum.push_back(1147);
	templateNum.push_back(1642);
	*/

	/*
	//fall over template1
	std::vector<int> templateNum;
	templateNum.push_back(154);
	templateNum.push_back(156);
	templateNum.push_back(159);
	templateNum.push_back(163);
	templateNum.push_back(166);
	templateNum.push_back(169);
	*/

	/*
	//fall over template2
	std::vector<int> templateNum;
	templateNum.push_back(1179);
	templateNum.push_back(1180);
	templateNum.push_back(1181);
	templateNum.push_back(1182);
	templateNum.push_back(1183);
	templateNum.push_back(1184);
	*/

	/*
	//fall back template1
	std::vector<int> templateNum;
	templateNum.push_back(744);
	templateNum.push_back(746);
	templateNum.push_back(748);
	templateNum.push_back(751);
	templateNum.push_back(753);
	templateNum.push_back(756);
	*/

	//fall back template1
	std::vector<int> templateNum;
	templateNum.push_back(1650);
	templateNum.push_back(1655);
	templateNum.push_back(1659);
	templateNum.push_back(1663);
	templateNum.push_back(1667);
	templateNum.push_back(1676);

	//set roi
	int roi_tl_x =0,roi_tl_y=0,roi_width=749,roi_height=720;		//roi in office left
	cv::Rect roi_half(roi_tl_x, roi_tl_y, roi_width, roi_height);	//roi in office left

	//input/output video
	std::string video = argv[1];
	cv::VideoCapture cap(video);
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}
	double rate = 25.0;  
	cv::Size videoSize(roi_width,roi_height);  
	cv::VideoWriter writer1("VideoTest_pose_only.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize); 
	cv::VideoWriter writer2("VideoTest_pose.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, videoSize); 

	int frameCount;

	cv::namedWindow("pose_only");
	cv::namedWindow("pose");

	cv::FileStorage fs(argv[2], cv::FileStorage::READ);
	cv::FileStorage fs1("poseCos.xml", cv::FileStorage::WRITE);
	fs1 << "people" << "[";

	if (!fs.isOpened())  
	{  
		std::cerr << "failed to open xml" << std::endl;  
	}
	else
	{
		std::cout << argv[2] << "successfully open" << std::endl;
	}  
	cv::FileNode frame_node = fs["frame"];//读取根节点
    cv::FileNodeIterator frame_node_iBeg = frame_node.begin(); //获取结构体数组迭代器
    cv::FileNodeIterator frame_node_iEnd = frame_node.end();

    for (; frame_node_iBeg != frame_node_iEnd; frame_node_iBeg++)
    {
    	if (cv::waitKey(1) == ' ')
			cv::waitKey();
		cv::Mat temp,frame,frameRoi;
		cap >> frame;
		frameCount = (*frame_node_iBeg)["frameNum"];
		if (!frame.data || cv::waitKey(1) == 27)        
		{
			std::cout<<"frame is unaviliable or press Esc, the video showing go to end"<<std::endl;
			break;
		}
		std::cout << "frame: " << frameCount << std::endl;	
		frameRoi = frame(roi_half);
		cv::Mat frameRoi_pose_only(cv::Size(roi_width, roi_height), CV_8UC3, cv::Scalar(0,0,0));

		std::vector<PosePoint> posePoints;		//some people

		cv::FileNode people_node = (*frame_node_iBeg)["people"];//读取根节点
		cv::FileNodeIterator people_node_iBeg = people_node.begin(); //获取结构体数组迭代器
		cv::FileNodeIterator people_node_iEnd = people_node.end();
		for (; people_node_iBeg != people_node_iEnd; people_node_iBeg++)
		{
			cv::FileNode joint_node = (*people_node_iBeg)["joint"];//读取根节点
			cv::FileNodeIterator joint_node_iBeg = joint_node.begin(); //获取结构体数组迭代器
			cv::FileNodeIterator joint_node_iEnd = joint_node.end();
			int count_joint = 0;
			PosePoint posePoint;		//some joint
			for (; joint_node_iBeg != joint_node_iEnd; joint_node_iBeg++)
			{
				cv::Point2d tmpCenter;
				tmpCenter.x= (double)(*joint_node_iBeg)["x_value"];
				tmpCenter.y= (double)(*joint_node_iBeg)["y_value"];
				posePoint.c[count_joint]= (double)(*joint_node_iBeg)["confidence"];
				posePoint.x[count_joint] = tmpCenter.x;
				posePoint.y[count_joint] = tmpCenter.y;
				if(posePoint.c[count_joint]!=0)
					cv::circle(frameRoi, tmpCenter, posePoint.c[count_joint]*5, cv::Scalar(255, 0, 0),-1);
				count_joint++;
			}
			posePoints.push_back(posePoint);
		}

		bool getTemplateFlag = true;
		for (std::vector<PosePoint>::iterator it = posePoints.begin(); it != posePoints.end();it++)
		{
			PoseCos poseCos;
			cv::Point2d tmpCenter_start,tmpCenter_end;
			int lineThickness = 3;
			tmpCenter_start.x= (*it).x[0];
			tmpCenter_start.y= (*it).y[0];
			tmpCenter_end.x= (*it).x[1];
			tmpCenter_end.y= (*it).y[1];
			if((*it).c[0]!=0 &&(*it).c[1]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,0);			
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,0);		
			}
				
			tmpCenter_start.x= (*it).x[1];
			tmpCenter_start.y= (*it).y[1];
			tmpCenter_end.x= (*it).x[2];
			tmpCenter_end.y= (*it).y[2];
			if((*it).c[1]!=0 &&(*it).c[2]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,1);	
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,1);	
			}

			tmpCenter_start.x= (*it).x[1];
			tmpCenter_start.y= (*it).y[1];
			tmpCenter_end.x= (*it).x[5];
			tmpCenter_end.y= (*it).y[5];
			if((*it).c[1]!=0 &&(*it).c[5]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);	
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,2);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,2);	
			}

			tmpCenter_start.x= (*it).x[1];
			tmpCenter_start.y= (*it).y[1];
			tmpCenter_end.x= (*it).x[8];
			tmpCenter_end.y= (*it).y[8];
			if((*it).c[1]!=0 &&(*it).c[8]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);	
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,3);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,3);	
			}

			tmpCenter_start.x= (*it).x[1];
			tmpCenter_start.y= (*it).y[1];
			tmpCenter_end.x= (*it).x[11];
			tmpCenter_end.y= (*it).y[11];
			if((*it).c[1]!=0 &&(*it).c[11]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);	
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,4);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,4);	
			}

			tmpCenter_start.x= (*it).x[2];
			tmpCenter_start.y= (*it).y[2];
			tmpCenter_end.x= (*it).x[3];
			tmpCenter_end.y= (*it).y[3];
			if((*it).c[2]!=0 &&(*it).c[3]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,5);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,5);	
			}

			tmpCenter_start.x= (*it).x[3];
			tmpCenter_start.y= (*it).y[3];
			tmpCenter_end.x= (*it).x[4];
			tmpCenter_end.y= (*it).y[4];
			if((*it).c[3]!=0 &&(*it).c[4]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,6);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,6);	
			}

			tmpCenter_start.x= (*it).x[5];
			tmpCenter_start.y= (*it).y[5];
			tmpCenter_end.x= (*it).x[6];
			tmpCenter_end.y= (*it).y[6];
			if((*it).c[5]!=0 &&(*it).c[6]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,7);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,7);	
			}

			tmpCenter_start.x= (*it).x[6];
			tmpCenter_start.y= (*it).y[6];
			tmpCenter_end.x= (*it).x[7];
			tmpCenter_end.y= (*it).y[7];
			if((*it).c[6]!=0 &&(*it).c[7]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,8);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,8);	
			}

			tmpCenter_start.x= (*it).x[8];
			tmpCenter_start.y= (*it).y[8];
			tmpCenter_end.x= (*it).x[9];
			tmpCenter_end.y= (*it).y[9];
			if((*it).c[8]!=0 &&(*it).c[9]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,9);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,9);	
			}

			tmpCenter_start.x= (*it).x[9];
			tmpCenter_start.y= (*it).y[9];
			tmpCenter_end.x= (*it).x[10];
			tmpCenter_end.y= (*it).y[10];
			if((*it).c[9]!=0 &&(*it).c[10]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,10);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,10);	
			}

			tmpCenter_start.x= (*it).x[11];
			tmpCenter_start.y= (*it).y[11];
			tmpCenter_end.x= (*it).x[12];
			tmpCenter_end.y= (*it).y[12];
			if((*it).c[11]!=0 &&(*it).c[12]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,11);
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,11);	
			}

			tmpCenter_start.x= (*it).x[12];
			tmpCenter_start.y= (*it).y[12];
			tmpCenter_end.x= (*it).x[13];
			tmpCenter_end.y= (*it).y[13];
			if((*it).c[12]!=0 &&(*it).c[13]!=0)
			{
				cv::line(frameRoi,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);	
				cv::line(frameRoi_pose_only,tmpCenter_start,tmpCenter_end,cv::Scalar(0,0,255),lineThickness,CV_AA);	
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,true,poseCos,12);									
			}
			else
			{
				computeCos(tmpCenter_start.x,tmpCenter_end.x,tmpCenter_start.y,tmpCenter_end.y,false,poseCos,12);	
			}

			if (getTemplateFlag)
			{	
				for(std::vector<int>::iterator itTemplate = templateNum.begin();itTemplate !=templateNum.end();itTemplate++)
				{
					if(frameCount == (*itTemplate))
					{
						fs1 << "{";
						fs1 << "frameNum" << frameCount;
						fs1 << "poseCos" << "[";
						for(int i=0;i<13;i++)
						{
							fs1 << poseCos.alpha[i];
							fs1 << poseCos.beta[i];
						}
						fs1 << "]";
						fs1 << "}";						
					}
				}				
			}
		}
			  		
		cv::Point disPos(5, 20);
		std::stringstream ssFrameCount;
		std::string sFrameCount;
		ssFrameCount << frameCount;
		ssFrameCount >> sFrameCount;
		sFrameCount = "frame:" + sFrameCount;
		cv::putText(frameRoi_pose_only, sFrameCount, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
		cv::putText(frameRoi, sFrameCount, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

		writer1 << frameRoi_pose_only;
		cv::imshow("pose_only", frameRoi_pose_only);
		writer2 << frameRoi;
		cv::imshow("pose", frameRoi);			
    }
    fs1 << "]";
    fs1.release();
    fs.release(); 
	of.flush();
	of.close();
	std::cout.rdbuf(coutBuf);
	std::cout << "Write Personal Information over..." << std::endl;
	return 0;
}

void computeCos(double x1,double x2,double y1,double y2,bool confidenceFlag,PoseCos &poseCos,int idx)
{
	if(confidenceFlag)
	{
		double distance = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
		poseCos.alpha[idx] = (x2-x1)/distance;
		poseCos.beta[idx] = (y2-y1)/distance;
	}
	else
	{
		poseCos.alpha[idx] = 0;
		poseCos.beta[idx] = 0;
	}
}