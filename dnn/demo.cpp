#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

vector<string> getOutputsNames(const dnn::Net& net)
{
	vector<string> names;
	vector<int> layerNum = net.getUnconnectedOutLayers();
	vector<string> layerName = net.getLayerNames();
	for ( ssize_t i=0;i<layerNum.size();i++)
		names.push_back(layerName[layerNum[i]-1]);
	return names;
}

int getMaxClassId(float *scores, int number)
{
	int maxId = 0;
	float maxScore = scores[0];
	for (ssize_t i=1;i<number;i++)
	{
		if (scores[i] > scores[maxId])
		{
			maxId = i;
			maxScore = scores[i];
		}
	}
	return maxId;
}

void postProcess(Mat& image, vector<Mat> outputs)
{
	vector<int> class_id;
	vector<float> confs;
	vector<Rect> bboxes;
	
	for (ssize_t i=0;i<outputs.size();i++)
	{
		float *ptr = (float*)outputs[i].data;
		for (ssize_t j=0;j<outputs[i].rows;j++)
		{
			float *data = ptr + j*outputs[i].cols;
			//obj confidence
			if (data[4] < 0.3)	continue;

			int maxId = getMaxClassId(data+5,outputs[i].cols - 5);
			class_id.push_back(maxId);
			confs.push_back(data[5+maxId]);
			int left = (int)((data[0] - data[2]*0.5 ) * image.cols);
			int top = (int)((data[1] - data[3]*0.5) *image.rows);
			int width = (int)(data[2] * image.cols);
			int height = (int)(data[3] * image.rows);
			Rect rect = Rect(left,top,width,height);
			bboxes.push_back(rect);
		}
	}
	vector<int> ids;
	dnn::NMSBoxes(bboxes,confs,0.3,0.45, ids);
	for (ssize_t i=0;i<ids.size();i++)
	{
		int id = ids[i];
		Rect rect = bboxes[id];
		rectangle(image, rect, Scalar(0,0,255),1,8);
	}
	resize(image,image,Size(704,576));
	imshow("result",image);
	waitKey(0);
}

int main(int argc, char** argv)
{
	if (argc != 4)	
	{
		cout << "usage:"<< argv[0] <<" cfg_path weights_path image_path" <<endl;
		return -1;
	}
	string cfg_path = argv[1];
	string weights_path = argv[2];
	string image_path = argv[3];

	//load network
	dnn::Net net = dnn::readNetFromDarknet(cfg_path,weights_path);
	net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(dnn::DNN_TARGET_CPU);
	
	//read image
	Mat image = imread(image_path);
	if(image.empty())	
	{
		cout << "read image failed!"<<endl;
		return -1;
	}
	
	Mat blob;
	dnn::blobFromImage(image,blob,1/255.,Size(416,416),Scalar(0,0,0),true,false);
	net.setInput(blob);

	vector<Mat> outputs;
	net.forward(outputs,getOutputsNames(net));

	postProcess(image, outputs);
	return 0;
}
