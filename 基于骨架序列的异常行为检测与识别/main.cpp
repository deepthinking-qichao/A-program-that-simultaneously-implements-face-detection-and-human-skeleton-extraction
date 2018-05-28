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

#define Min(a,b) (a<b?a:b) 

struct PoseCos
{
	double directionCos[26];
};

//abnormal behavior structure
struct PosePoint
{
	double x[18];
	double y[18];
	double c[18];
	int idx;
	int abnormalActionCount;
	int normalActionCount;
	std::vector<PoseCos> abnormalPoseCosSet;
	std::string abnormalAction;
};

struct ShowAbnormalAction
{
	int idx;
	int restFrameNum;
	cv::Point disPos;
	std::string abnormalAction;
};

void computeCos(double x1,double x2,double y1,double y2,bool confidenceFlag,PoseCos &poseCos,int idx);
void ReadTemplate(const cv::FileStorage &fs, std::vector<PoseCos> &poseCos,std::string xmlName);
double ComputeCosDistance(std::vector<PoseCos> abnormalPoseCosSet, std::vector<PoseCos> templatePoseCosSet);
double TwoVectorDistance(PoseCos a,PoseCos b);

int main(int argc, char** argv)
{
	std::streambuf* coutBuf = std::cout.rdbuf();
	std::ofstream of("out.txt");
	std::streambuf* fileBuf = of.rdbuf();
	std::cout.rdbuf(fileBuf);

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
	int PpIdx = 0;
	int abnormalFrameNumThreshold = 3;
	int normalFrameNumThreshold = 1;
	int showAbnormalActionTime = 25;
	int AbnormalActionRestTime = 25;

	cv::namedWindow("pose_only");
	cv::namedWindow("pose");

	cv::FileStorage fs(argv[2], cv::FileStorage::READ);
	cv::FileStorage fs1("poseCosAb.xml", cv::FileStorage::READ);
	cv::FileStorage fsFO1("poseCosFO1.xml", cv::FileStorage::READ);
	cv::FileStorage fsFO2("poseCosFO2.xml", cv::FileStorage::READ);
	cv::FileStorage fsFB1("poseCosFB1.xml", cv::FileStorage::READ);
	cv::FileStorage fsFB2("poseCosFB2.xml", cv::FileStorage::READ);

	//read abnormal pose cos vector
	std::vector<PoseCos> poseCosAb;
	ReadTemplate(fs1, poseCosAb, "poseCosAb.xml");

    //read fall over one pose cos vector
    std::vector<PoseCos> poseCosFO1;
	ReadTemplate(fsFO1, poseCosFO1, "poseCosFO1.xml");

    //read fall over two pose cos vector
    std::vector<PoseCos> poseCosFO2;
	ReadTemplate(fsFO2, poseCosFO2, "poseCosFO2.xml");

    //read fall back one pose cos vector
    std::vector<PoseCos> poseCosFB1;
	ReadTemplate(fsFB1, poseCosFB1, "poseCosFB1.xml");

    //read fall back two pose cos vector
    std::vector<PoseCos> poseCosFB2;
	ReadTemplate(fsFB2, poseCosFB2, "poseCosFB2.xml");
	
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

	std::vector<PosePoint> posePointsPres;
	std::vector<ShowAbnormalAction> showAbnormalActionSet;
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
		//std::cout << "idx: " << PpIdx << std::endl;
		frameRoi = frame(roi_half);
		cv::Mat frameRoi_pose_only(cv::Size(roi_width, roi_height), CV_8UC3, cv::Scalar(0,0,0));

		std::vector<PosePoint> posePoints;		//some people

		cv::FileNode people_node = (*frame_node_iBeg)["people"];//读取根节点
		cv::FileNodeIterator people_node_iBeg = people_node.begin(); //获取结构体数组迭代器
		cv::FileNodeIterator people_node_iEnd = people_node.end();

		//Match the passenger nose of the current frame and the previous frame to determine whether it is the same person. 
		//If not, initialize the abnormal behavior structure of the current frame passenger.		
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
			if (frameCount == 0)
			{
				posePoint.idx = PpIdx++;
				posePoint.abnormalActionCount = 0;
				posePoint.normalActionCount = 0;
				posePoint.abnormalPoseCosSet.clear();
				posePoint.abnormalAction = "";
			}
			else
			{
				//double distThreshold = 30;
				double distMin = 30;
				double x2 = posePoint.x[0];
				double y2 = posePoint.y[0];
				bool matchFlag = false; 
				for(std::vector<PosePoint>::iterator it = posePointsPres.begin(); it != posePointsPres.end();it++)
				{
					double x1 = (*it).x[0];
					double y1 = (*it).y[0];
					double distance = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
					std::cout << "people distance between two frame is " << distance << std::endl;
					if (distance < distMin)		
					{
						distMin = distance;
						posePoint.idx = (*it).idx;
						posePoint.abnormalActionCount = (*it).abnormalActionCount;
						//posePoint.normalActionCount = (*it).abnormalActionCount;
						posePoint.normalActionCount = (*it).normalActionCount;
						posePoint.abnormalPoseCosSet.clear();
						posePoint.abnormalPoseCosSet = (*it).abnormalPoseCosSet;
						posePoint.abnormalAction = (*it).abnormalAction;
						matchFlag = true;
					}						
				}
				
				if (matchFlag == false)
				{
					posePoint.idx = PpIdx++;
					posePoint.abnormalActionCount = 0;
					posePoint.normalActionCount = 0;
					posePoint.abnormalPoseCosSet.clear();
					posePoint.abnormalAction = "";
				}

				std::cout << "the minimum people distance between two frame is " << distMin << std::endl;
				std::cout << "people id is " << posePoint.idx << std::endl;				
			}
			posePoints.push_back(posePoint);
		}

		//If the previous frame passenger does not match the current frame and the number of abnormal frames of the previous frame passenger is not less than 6, 
		//the abnormal behavior of the passenger is identified
		for(std::vector<PosePoint>::iterator it = posePointsPres.begin(); it != posePointsPres.end();it++)
		{
			bool matchFlag = false;
			for (std::vector<PosePoint>::iterator it1 = posePoints.begin(); it1 != posePoints.end();it1++)
			{
				if((*it).idx == (*it1).idx)
				{
					matchFlag = true;
					break;
				}
			}
			if (!matchFlag)
			{
				//if ((*it).abnormalPoseCosSet.size()>=6 && (*it).abnormalPoseCosSet.size()<=AbnormalActionRestTime)
				if ((*it).abnormalPoseCosSet.size()>=6)
				{
					(*it).abnormalAction.clear();
					std::cout<<"previous picture:compare with abnormal action template"<<std::endl;
					double minDistance = 10000;
					double tempDistance;
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFO1);
					std::cout<<"FO1:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall over";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFO2);
					std::cout<<" FO2:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall over";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFB1);
					std::cout<<" FB1:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall back";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFB2);
					std::cout<<" FB2:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall back";
					}

					if(!(*it).abnormalAction.empty())
					{
						std::cout<<" the result is "<<(*it).abnormalAction<<std::endl;
						ShowAbnormalAction showAbnormalActionTemp;
						showAbnormalActionTemp.idx = (*it).idx;
						showAbnormalActionTemp.restFrameNum = showAbnormalActionTime;
						showAbnormalActionTemp.disPos.x = (*it).x[0];	
						showAbnormalActionTemp.disPos.y = (*it).y[0];
						showAbnormalActionTemp.abnormalAction = (*it).abnormalAction;
						showAbnormalActionSet.push_back(showAbnormalActionTemp);
					}	
				}		
			}
		}

		//Display skeleton information of all passengers in the current frame
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

			//Detect abnormal behavior
			double cosDistMin = 10000;
			std::cout << "cos distance for each person is "; 
			for(std::vector<PoseCos>::iterator itPoseCosAb = poseCosAb.begin();itPoseCosAb!=poseCosAb.end();itPoseCosAb++)
			{
				double cosDist =0;
				for(int i=0;i<26;i++)
					cosDist += pow((*itPoseCosAb).directionCos[i]-poseCos.directionCos[i],2);
				cosDist = sqrt(cosDist);
				std::cout << cosDist << " ";
				if (cosDist < cosDistMin)
					cosDistMin = cosDist;			
			}
			std::cout << std::endl;
			std::cout << "the minimum cos distance is "<< cosDistMin << std::endl;

			//If there is abnormal behavior and the number of abnormal frames is greater than a certain threshold, identify abnormal behavior
			if(cosDistMin > 1.25)
			{
				(*it).abnormalActionCount++;
				(*it).normalActionCount = 0;
				if((*it).abnormalPoseCosSet.size() > AbnormalActionRestTime)
				{
					std::cout<<"picture abnormal last out of time :compare with abnormal action template"<<std::endl;
					double minDistance = 10000;
					double tempDistance;
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFO1);
					std::cout<<"FO1:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall over";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFO2);
					std::cout<<" FO2:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall over";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFB1);
					std::cout<<" FB1:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall back";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFB2);
					std::cout<<" FB2:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall back";
					}
					if(!(*it).abnormalAction.empty())
					{
						std::cout<<" the result is "<<(*it).abnormalAction<<std::endl;
						cv::Point disPos((*it).x[0], (*it).y[0]+20);
						std::stringstream ssFrameAb;			
						std::string sFrameAb;		
						ssFrameAb << (*it).idx;
						ssFrameAb >> sFrameAb;			
						sFrameAb = sFrameAb + ":" + (*it).abnormalAction;
						cv::putText(frameRoi, sFrameAb, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
					}
				}				
			}
			else
			{
				(*it).abnormalActionCount = 0;
				(*it).normalActionCount++;
			}

			//The number of abnormal frames in the current frame exceeds a certain threshold, indicating an abnormal state
			if((*it).abnormalActionCount > abnormalFrameNumThreshold)
			{
				cv::Point disPos((*it).x[0], (*it).y[0]);
				std::stringstream ssFrameAb;			
				std::string sFrameAb;		
				ssFrameAb << (*it).idx;
				ssFrameAb >> sFrameAb;			
				sFrameAb += ":Abnormal";		
				cv::putText(frameRoi, sFrameAb, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
				(*it).abnormalPoseCosSet.push_back(poseCos);
			}

			//The number of normal frames of passengers in the current frame is greater than a certain number of frames. 
			//The reserved abnormal sequence is identified and the abnormal sequence is cleared.
			if((*it).normalActionCount > normalFrameNumThreshold)
			{
				if ((*it).abnormalPoseCosSet.size()>=6)
				{
					(*it).abnormalAction.clear();
					std::cout<<"picture people turn to normal state:compare with abnormal action template" <<std::endl;
					double minDistance = 10000;
					double tempDistance;
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFO1);
					std::cout<<"FO1:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall over";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFO2);
					std::cout<<" FO2:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall over";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFB1);
					std::cout<<" FB1:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall back";
					}
					tempDistance = ComputeCosDistance((*it).abnormalPoseCosSet,poseCosFB2);
					std::cout<<" FB2:" << tempDistance;
					if(tempDistance < minDistance)
					{
						minDistance = tempDistance;
						(*it).abnormalAction = "fall back";
					}

					if(!(*it).abnormalAction.empty())
					{
						std::cout<<" the result is "<<(*it).abnormalAction<<std::endl;
						ShowAbnormalAction showAbnormalActionTemp;
						showAbnormalActionTemp.idx = (*it).idx;
						showAbnormalActionTemp.restFrameNum = showAbnormalActionTime;
						showAbnormalActionTemp.disPos.x = (*it).x[0];	
						showAbnormalActionTemp.disPos.y = (*it).y[0];
						showAbnormalActionTemp.abnormalAction = (*it).abnormalAction;
						showAbnormalActionSet.push_back(showAbnormalActionTemp);
					}
				}
				
				(*it).abnormalPoseCosSet.clear();
			}
		}

		//Display abnormal information before elimination and control the display time to a certain length
		for (std::vector<ShowAbnormalAction>::iterator it = showAbnormalActionSet.begin(); it != showAbnormalActionSet.end();)
        {
        	if((*it).restFrameNum<0)
        		it=showAbnormalActionSet.erase(it);
        	else
        	{
        		cv::Point disPos = (*it).disPos;
        		disPos.y +=20;

				std::stringstream ssFrameAb;			
				std::string sFrameAb;		
				ssFrameAb << (*it).idx;
				ssFrameAb >> sFrameAb;			
				sFrameAb = sFrameAb + ":" + (*it).abnormalAction;
				cv::putText(frameRoi, sFrameAb, disPos, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
				(*it).restFrameNum--;
        		it++;
        	}
        }

		posePointsPres.clear();
		posePointsPres = posePoints;
			  		
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
   
    fsFO1.release();
    fsFO2.release();
    fsFB1.release();
    fsFB2.release();
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
		idx *= 2;
		double distance = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
		poseCos.directionCos[idx] = (x2-x1)/distance;
		poseCos.directionCos[idx+1] = (y2-y1)/distance;
	}
	else
	{
		poseCos.directionCos[idx]  = 0;
		poseCos.directionCos[idx+1] = 0;
	}
}

void ReadTemplate(const cv::FileStorage &fs, std::vector<PoseCos> &poseCos,std::string xmlName)
{
	if (!fs.isOpened())  
	{  
		std::cerr << "failed to open " << xmlName << std::endl;  
	}
	else
	{
		std::cout << xmlName << " successfully open" << std::endl;
	}  
	cv::FileNode template_people_node = fs["people"];
	cv::FileNodeIterator template_people_node_iBeg = template_people_node.begin(); 
    cv::FileNodeIterator template_people_node_iEnd = template_people_node.end();
    for (; template_people_node_iBeg != template_people_node_iEnd; template_people_node_iBeg++)
    {
    	PoseCos poseCosTemp;
    	cv::FileNode template_poseCos_node = (*template_people_node_iBeg)["poseCos"];
    	cv::FileNodeIterator template_poseCos_node_iBeg = template_poseCos_node.begin(); 
    	cv::FileNodeIterator template_poseCos_node_iEnd = template_poseCos_node.end();
    	int count =0;
    	for (; template_poseCos_node_iBeg != template_poseCos_node_iEnd; template_poseCos_node_iBeg++)
    	{
    		poseCosTemp.directionCos[count++] = (double)(*template_poseCos_node_iBeg);
    	}
    	poseCos.push_back(poseCosTemp);//write here
    	for(int i=0;i<26;i++)
    		std::cout<<poseCosTemp.directionCos[i]<<" ";
    	std::cout<<std::endl;
    }
}

//computing distance using DTW algorithm
double ComputeCosDistance(std::vector<PoseCos> abnormalPoseCosSet, std::vector<PoseCos> templatePoseCosSet)
{
	int m = abnormalPoseCosSet.size();
	int n = templatePoseCosSet.size();
	double distance[m+1][n+1];
	double output[m+1][n+1];
	memset(distance,0,sizeof(distance)); 
	for(int i=1; i<=m; i++)	
		output[i][0] = 10000;
	for(int j=1; j<=n; j++)	
		output[0][j] = 10000;
	output[0][0] = 0;

	for(int i=1;i<=m;i++)
		for(int j=1;j<=n;j++)
		{
			distance[i][j]=TwoVectorDistance(abnormalPoseCosSet[i-1],templatePoseCosSet[j-1]);
			output[i][j]=Min(Min(output[i-1][j-1],output[i][j-1]),output[i-1][j])+distance[i][j]; 
		}

	//for(int)

	return output[m][n];
}

double TwoVectorDistance(PoseCos a,PoseCos b)
{
	double distance;
	for(int i=0;i<26;i++)
	{
		distance += pow(a.directionCos[i]-b.directionCos[i],2);
	}
	distance = sqrt(distance);
	return distance;
}