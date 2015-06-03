#include "match.h"

void BubbleSort(double* pData, int* index, int iCount)
{
	double dTemp;
	int iTemp;

	for (int i = 1; i < iCount; i++)
		for (int j = 0; j < iCount - i; j++)
			if (pData[j] > pData[j + 1])
			{
				dTemp = pData[j];
				iTemp = index[j];
				pData[j] = pData[j + 1];
				index[j] = index[j + 1];
				pData[j + 1] = dTemp;
				index[j + 1] = iTemp;
			}
}

void drawArrowColor(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness = 1, int lineType = 8)
{
	const double PI = 3.1415926;
	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
}


int main()
{
	Mat Img[3][4][2];
	// Read test Images
	// For Img[i][j][k], i means database, j means movement, k means frame
	//     i: 0 for realfly, 1 for airbrone, 2 for libviso
	//     j: 0 for scale, 1 for rotation, 2 for translation, 3 for projection
	//     k: 0 for previous frame, 1 for current frame
	// For libviso, images are taken by a carboard camera, so only scale and projection can be tested
	Img[0][0][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_scale_pre.jpg");
	Img[0][0][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_scale_cur.jpg");
	Img[0][1][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_rotation_pre.jpg");
	Img[0][1][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_rotation_cur.jpg");
	Img[0][2][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_trans_pre.jpg");
	Img[0][2][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_trans_cur.jpg");
	Img[0][3][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_proj_pre.jpg");
	Img[0][3][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_realfly_proj_cur.jpg");
	Img[1][0][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_scale_pre.jpg");
	Img[1][0][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_scale_cur.jpg");
	Img[1][1][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_rotation_pre.jpg");
	Img[1][1][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_rotation_cur.jpg");
	Img[1][2][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_trans_pre.jpg");
	Img[1][2][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_trans_cur.jpg");
	Img[1][3][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_proj_pre.jpg");
	Img[1][3][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_airbrone_proj_cur.jpg");
	Img[2][0][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_scale_pre.png");
	Img[2][0][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_scale_cur.png");
	Img[2][1][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_scale_pre.png");
	Img[2][1][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_scale_cur.png");
	Img[2][2][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_scale_pre.png");
	Img[2][2][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_scale_cur.png");
	Img[2][3][0] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_proj_pre.png");
	Img[2][3][1] = imread("F:\\Works\\MyProgram\\VisualStudio\\InterframeMatchAlgorithms\\pic_libviso_proj_cur.png");

	for (int x = 0; x < 3; x++)
	for (int y = 0; y < 4; y++)
	{
		//Output image name
		cout << "Images of ";
		switch (y){
		case 0: 
			cout << "scale ";
			break;
		case 1:
			cout << "rotation ";
			break;
		case 2:
			cout << "transposition ";
			break;
		case 3:
			cout << "projection ";
			break;
		}
		cout << "in ";
		switch (x){
		case 0:
			cout << "realfly data.";
			break;
		case 1:
			cout << "airbrone database.";
			break;
		case 2:
			cout << "libviso database.";
			break;
		}
		cout << endl;

		// Feature matching
		vector<KeyPoint> KeyPointsP;			//previous feature keypoints
		vector<KeyPoint> KeyPointsC;			//current feature keypoints
		SurfFeatureDetector Detector(2500);		//surf detector
		Detector.detect(Img[x][y][0], KeyPointsP);		//detect feature points in previous image
		Detector.detect(Img[x][y][1], KeyPointsC);		//detect feature points in current image
		//获取特征点描述子
		SurfDescriptorExtractor Extractor;
		Mat DescriptorsP, DescriptorsC;
		Extractor.compute(Img[x][y][0], KeyPointsP, DescriptorsP);
		Extractor.compute(Img[x][y][1], KeyPointsC, DescriptorsC);
		//特征匹配
		BruteForceMatcher<L2<float> > Matcher;
		vector<DMatch> Matches;
		Matcher.match(DescriptorsP, DescriptorsC, Matches);
		cout << "Raw SURF matches: " << Matches.size() << endl;

		//将KeyPoint转化为Mat以正常调用findFundamentalMat
		int ptCount = (int)Matches.size();
		Mat ptP(ptCount, 2, CV_32F);		//coordinate of points in previous frame
		Mat ptC(ptCount, 2, CV_32F);		//coordinate of points in current frame
		for (int i = 0; i < ptCount; i++)
		{
			Point2f pt;							//temp points in the format of point2f
			pt = KeyPointsP[Matches[i].queryIdx].pt;
			ptP.at<float>(i, 0) = pt.x;
			ptP.at<float>(i, 1) = pt.y;
			pt = KeyPointsC[Matches[i].trainIdx].pt;
			ptC.at<float>(i, 0) = pt.x;
			ptC.at<float>(i, 1) = pt.y;
		}
		//计算基础矩阵
		Mat FMatrix;
		vector<uchar> RANSACstatus;
		FMatrix = findFundamentalMat(ptP, ptC, RANSACstatus, FM_RANSAC);
		//获取野点Outliner数目
		int OutlinerCount = 0;
		for (int i = 0; i < ptCount; i++)
			if (RANSACstatus[i] == 0)		//outliner when statue is 0
				OutlinerCount++;
		//计算内点Inliner
		vector<Point2f> InlierP;			//inliners in previous frame
		vector<Point2f> InlierC;			//inliners in current frame
		vector<DMatch> InlierMatches;
		int InlinerCount = ptCount - OutlinerCount;
		InlierMatches.resize(InlinerCount);
		InlierP.resize(InlinerCount);
		InlierC.resize(InlinerCount);
		InlinerCount = 0;
		for (int i = 0; i<ptCount; i++)
			if (RANSACstatus[i] != 0)
			{
				InlierP[InlinerCount].x = ptP.at<float>(i, 0);
				InlierP[InlinerCount].y = ptP.at<float>(i, 1);
				InlierC[InlinerCount].x = ptC.at<float>(i, 0);
				InlierC[InlinerCount].y = ptC.at<float>(i, 1);
				InlierMatches[InlinerCount].queryIdx = InlinerCount;
				InlierMatches[InlinerCount].trainIdx = InlinerCount;
				InlinerCount++;
			}
		cout << "SURF matches after RANSAC: " << InlinerCount << endl;

		int BetterMatchCount = 0;
		vector<Point2f> BetterMatchP;			//better match in previous frame
		vector<Point2f> BetterMatchC;			//better match in current frame
		vector<DMatch> BetterMatches;
		BetterMatchP.resize(InlinerCount);
		BetterMatchC.resize(InlinerCount);
		BetterMatches.resize(InlinerCount);
		for (int i = 0; i < InlinerCount; i++)
		{
			int distancetoorigin = sqrt((InlierP[i].x - 277) * (InlierP[i].x - 277) + (InlierP[i].y - 185) * (InlierP[i].y - 185));
			//if (distancetoorigin < 150)
			{
				BetterMatchP[BetterMatchCount] = InlierP[i];
				BetterMatchC[BetterMatchCount] = InlierC[i];
				BetterMatches[BetterMatchCount].queryIdx = BetterMatchCount;
				BetterMatches[BetterMatchCount].trainIdx = BetterMatchCount;
				BetterMatchCount++;
			}
		}
		BetterMatchP.resize(BetterMatchCount);
		BetterMatchC.resize(BetterMatchCount);
		BetterMatches.resize(BetterMatchCount);
		cout << "SURF matches after circle: " << BetterMatches.size() << endl;

		//判断各特征光流方向以筛选之
		double *MatchDirection = new double[BetterMatchCount];
		int *MatchIndex = new int[BetterMatchCount];
		for (int i = 0; i < BetterMatchCount; i++){
			MatchDirection[i] = (double)(BetterMatchC[i].y + Img[x][y][1].rows - BetterMatchP[i].y) / (double)(BetterMatchC[i].x + Img[x][y][1].cols - BetterMatchP[i].x);
			MatchIndex[i] = i;
		}
		BubbleSort(MatchDirection, MatchIndex, BetterMatchCount);
		int CutNum = BetterMatchCount / 10;
		int FinalNum = BetterMatchCount - CutNum * 2;
		for (int i = 0; i < FinalNum; i++)
			MatchIndex[i] = MatchIndex[i + CutNum];
		vector<Point2f> FinalMatchP;			//better match in previous frame
		vector<Point2f> FinalMatchC;			//better match in current frame
		vector<DMatch> FinalMatches;
		FinalMatchP.resize(FinalNum);
		FinalMatchC.resize(FinalNum);
		FinalMatches.resize(FinalNum);
		for (int i = 0; i < FinalNum; i++){
			FinalMatchP[i] = BetterMatchP[MatchIndex[i]];
			FinalMatchC[i] = BetterMatchC[MatchIndex[i]];
			FinalMatches[i].queryIdx = i;
			FinalMatches[i].trainIdx = i;
		}
		cout << "SURF matches final: " << FinalMatches.size() << endl;

		//显示匹配结果
		Mat img_matches;
		char s_image_name[256];
		// Show Raw Matches
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < Matches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, KeyPointsP[Matches[i].queryIdx].pt, KeyPointsC[Matches[i].trainIdx].pt, 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_surf_%d_%d_raw.jpg", x, y);
		imwrite(s_image_name, img_matches);
		// Show Matches after Ransac
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < InlierMatches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, InlierP[i], InlierC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_surf_%d_%d_ransac.jpg", x, y);
		imwrite(s_image_name, img_matches);
		// Show better matches
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < BetterMatches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, BetterMatchP[i], BetterMatchC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_surf_%d_%d_better.jpg", x, y);
		imwrite(s_image_name, img_matches);
		// Show final matches
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < FinalMatches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, FinalMatchP[i], FinalMatchC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_surf_%d_%d_final.jpg", x, y);
		imwrite(s_image_name, img_matches);
	}


	for (int x = 0; x < 3; x++)
	for (int y = 0; y < 4; y++)
	{
		//Output image name
		cout << "Images of ";
		switch (y){
		case 0:
			cout << "scale ";
			break;
		case 1:
			cout << "rotation ";
			break;
		case 2:
			cout << "transposition ";
			break;
		case 3:
			cout << "projection ";
			break;
		}
		cout << "in ";
		switch (x){
		case 0:
			cout << "realfly data.";
			break;
		case 1:
			cout << "airbrone database.";
			break;
		case 2:
			cout << "libviso database.";
			break;
		}
		cout << endl;

		Mat imgptemp, imgctemp;
		cv::cvtColor(Img[x][y][0], imgptemp, CV_BGR2GRAY);
		cv::cvtColor(Img[x][y][1], imgctemp, CV_BGR2GRAY);
		CvMat imgp = imgptemp;
		CvMat imgc = imgctemp;
		// OF

		CvSize img_sz = cvGetSize(&imgp);
		const int win_size = 10;
				//get good features 
				IplImage* img_eig = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
				IplImage* img_temp = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
				int corner_count = 500;
				CvPoint2D32f*  features_prev = new CvPoint2D32f[500];

				cvGoodFeaturesToTrack(
					&imgp,
					img_eig,
					img_temp,
					features_prev,
					&corner_count,
					0.01,
					5.0,
					0,
					3,
					0,
					0.4
					);

				cvFindCornerSubPix(
					&imgp,
					features_prev,
					corner_count,
					cvSize(win_size, win_size),
					cvSize(-1, -1),
					cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.03)
					);

				// L-K 
				char feature_found[500];
				float feature_errors[500];

				//CvSize pyr_sz = cvSize(frame->width + 8, frame->height / 3);

				IplImage* pyr_prev = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
				IplImage* pyr_cur = cvCreateImage(img_sz, IPL_DEPTH_32F, 1);
				CvPoint2D32f*  features_cur = new CvPoint2D32f[500];

				cvCalcOpticalFlowPyrLK(
					&imgp,
					&imgc,
					pyr_prev,
					pyr_cur,
					features_prev,
					features_cur,
					corner_count,
					cvSize(win_size, win_size),
					5,
					feature_found,
					feature_errors,
					cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.3),
					0
					);
			vector<Point2f> KeyPointsP;			//inliners in previous frame
			vector<Point2f> KeyPointsC;			//inliners in current frame
			KeyPointsP.resize(500);
			KeyPointsC.resize(500);
			int KeyPointsnum = 0;
				for (int i = 0; i < corner_count; i++)
				{
					if (0 == feature_found[i] || feature_errors[i] > 550)
					{
						//printf("error is %f \n", feature_errors[i]);
						continue;
					}
					KeyPointsP[KeyPointsnum] = features_prev[i];
					KeyPointsC[KeyPointsnum] = features_cur[i];
					KeyPointsnum++;
					//cvLine(img_res, pt_prev, pt_cur, CV_RGB(255, 0, 0), 2);
				}
				KeyPointsP.resize(KeyPointsnum);
				KeyPointsC.resize(KeyPointsnum);
				cout << "OF raw matches: " << KeyPointsnum << endl;


		//将KeyPoint转化为Mat以正常调用findFundamentalMat
		int ptCount = (int)KeyPointsC.size();
		Mat ptP(ptCount, 2, CV_32F);		//coordinate of points in previous frame
		Mat ptC(ptCount, 2, CV_32F);		//coordinate of points in current frame
		for (int i = 0; i < ptCount; i++)
		{
			Point2f pt;							//temp points in the format of point2f
			pt = KeyPointsP[i];
			ptP.at<float>(i, 0) = pt.x;
			ptP.at<float>(i, 1) = pt.y;
			pt = KeyPointsC[i];
			ptC.at<float>(i, 0) = pt.x;
			ptC.at<float>(i, 1) = pt.y;
		}
		//计算基础矩阵
		Mat FMatrix;
		vector<uchar> RANSACstatus;
		FMatrix = findFundamentalMat(ptP, ptC, RANSACstatus, FM_RANSAC);
		//获取野点Outliner数目
		int OutlinerCount = 0;
		for (int i = 0; i < ptCount; i++)
		if (RANSACstatus[i] == 0)		//outliner when statue is 0
			OutlinerCount++;
		//计算内点Inliner
		vector<Point2f> InlierP;			//inliners in previous frame
		vector<Point2f> InlierC;			//inliners in current frame
		vector<DMatch> InlierMatches;
		int InlinerCount = ptCount - OutlinerCount;
		InlierMatches.resize(InlinerCount);
		InlierP.resize(InlinerCount);
		InlierC.resize(InlinerCount);
		InlinerCount = 0;
		for (int i = 0; i<ptCount; i++)
		if (RANSACstatus[i] != 0)
		{
			InlierP[InlinerCount].x = ptP.at<float>(i, 0);
			InlierP[InlinerCount].y = ptP.at<float>(i, 1);
			InlierC[InlinerCount].x = ptC.at<float>(i, 0);
			InlierC[InlinerCount].y = ptC.at<float>(i, 1);
			InlierMatches[InlinerCount].queryIdx = InlinerCount;
			InlierMatches[InlinerCount].trainIdx = InlinerCount;
			InlinerCount++;
		}
		cout << "OF matches after RANSAC: " << InlinerCount << endl;

		int BetterMatchCount = 0;
		vector<Point2f> BetterMatchP;			//better match in previous frame
		vector<Point2f> BetterMatchC;			//better match in current frame
		vector<DMatch> BetterMatches;
		BetterMatchP.resize(InlinerCount);
		BetterMatchC.resize(InlinerCount);
		BetterMatches.resize(InlinerCount);
		for (int i = 0; i < InlinerCount; i++)
		{
			int distancetoorigin = sqrt((InlierP[i].x - 277) * (InlierP[i].x - 277) + (InlierP[i].y - 185) * (InlierP[i].y - 185));
			//if (distancetoorigin < 150)
			{
				BetterMatchP[BetterMatchCount] = InlierP[i];
				BetterMatchC[BetterMatchCount] = InlierC[i];
				BetterMatches[BetterMatchCount].queryIdx = BetterMatchCount;
				BetterMatches[BetterMatchCount].trainIdx = BetterMatchCount;
				BetterMatchCount++;
			}
		}
		BetterMatchP.resize(BetterMatchCount);
		BetterMatchC.resize(BetterMatchCount);
		BetterMatches.resize(BetterMatchCount);
		cout << "OF matches after circle: " << BetterMatches.size() << endl;

		//判断各特征光流方向以筛选之
		double *MatchDirection = new double[BetterMatchCount];
		int *MatchIndex = new int[BetterMatchCount];
		for (int i = 0; i < BetterMatchCount; i++){
			MatchDirection[i] = (double)(BetterMatchC[i].y + Img[x][y][1].rows - BetterMatchP[i].y) / (double)(BetterMatchC[i].x + Img[x][y][1].cols - BetterMatchP[i].x);
			MatchIndex[i] = i;
		}
		BubbleSort(MatchDirection, MatchIndex, BetterMatchCount);
		int CutNum = BetterMatchCount / 10;
		int FinalNum = BetterMatchCount - CutNum * 2;
		for (int i = 0; i < FinalNum; i++)
			MatchIndex[i] = MatchIndex[i + CutNum];
		vector<Point2f> FinalMatchP;			//better match in previous frame
		vector<Point2f> FinalMatchC;			//better match in current frame
		vector<DMatch> FinalMatches;
		FinalMatchP.resize(FinalNum);
		FinalMatchC.resize(FinalNum);
		FinalMatches.resize(FinalNum);
		for (int i = 0; i < FinalNum; i++){
			FinalMatchP[i] = BetterMatchP[MatchIndex[i]];
			FinalMatchC[i] = BetterMatchC[MatchIndex[i]];
			FinalMatches[i].queryIdx = i;
			FinalMatches[i].trainIdx = i;
		}
		cout << "OF matches final: " << FinalMatches.size() << endl;

		//显示匹配结果
		Mat img_matches;
		char s_image_name[256];
		// Show Raw Matches
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < KeyPointsC.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, KeyPointsP[i], KeyPointsC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_of_%d_%d_raw.jpg", x, y);
		imwrite(s_image_name, img_matches);
		// Show Matches after Ransac
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < InlierMatches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, InlierP[i], InlierC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_of_%d_%d_ransac.jpg", x, y);
		imwrite(s_image_name, img_matches);
		// Show better matches
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < BetterMatches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, BetterMatchP[i], BetterMatchC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_of_%d_%d_better.jpg", x, y);
		imwrite(s_image_name, img_matches);
		// Show final matches
		Img[x][y][0].copyTo(img_matches);
		for (int i = 0; i < FinalMatches.size(); i++)
		{
			Scalar lineColor(0, 0, 255);
			drawArrowColor(img_matches, FinalMatchP[i], FinalMatchC[i], 10, 45, lineColor, 1, 4);
		}
		sprintf_s(s_image_name, "result_of_%d_%d_final.jpg", x, y);
		imwrite(s_image_name, img_matches);
	}
	system("PAUSE");
}