#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "DetectRegions.h"
#include "OCR.h"
#include <fstream>

using namespace std;
using namespace cv;



int main()
{

    cout << "OpenCV Automatic Number Plate Recognition\n";
    string filename;
    Mat input_image;

        filename="TP1_computer_vision\\2715DTZ.JPG";
        input_image=imread(filename,1);

    //Detect posibles plate regions
    DetectRegions detectRegions;
    detectRegions.setFilename("2715DTZ");
    detectRegions.saveRegions=true;
    detectRegions.showSteps=false;
    vector<Plate> posible_regions= detectRegions.run( input_image );
    string filename1 = "result";


   //SVM for each plate region to get valid car plates
    //Read file storage.
    FileStorage fs;
    fs.open("TP1_computer_vision\\SVM.XML", FileStorage::READ);
    Mat SVM_TrainingData;
    Mat SVM_Classes;
    fs["TrainingData"] >> SVM_TrainingData;
    fs["classes"] >> SVM_Classes;
    //Set SVM params
    CvSVMParams SVM_params;
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;
    SVM_params.degree = 0;
    SVM_params.gamma = 1;
    SVM_params.coef0 = 0;
    SVM_params.C = 1;
    SVM_params.nu = 0;
    SVM_params.p = 0;
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    //Train SVM
    CvSVM svmClassifier(SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params);

    //For each possible plate, classify with svm if it's a plate or no
    vector<Plate> plates;
    for(int i=0; i< (int)posible_regions.size(); i++)
    {
        Mat img=posible_regions[i].plateImg;
        Mat p= img.reshape(1, 1);
        p.convertTo(p, CV_32FC1);

        int response = (int)svmClassifier.predict( p );
        if(response==1){

                stringstream ss(stringstream::in | stringstream::out);
                ss << ".\\output\\predicted\\" << filename1 <<".JPG";
                imwrite(ss.str(), posible_regions[i].plateImg);

                plates.push_back(posible_regions[i]);
    }}



	//imshow("plate",plates);
    cout << "Num plates detected: " << plates.size() << "\n\n\n";







//For each plate detected, recognize it with OCR
    OCR ocr("TP1_computer_vision\\OCR.XML");
    ocr.filename="2715DTZ";
    ocr.saveSegments=false;
    ocr.DEBUG=false;

    for(int i=0; i< (int) plates.size(); i++){
        Plate plate=plates[i];

        string plateNumber=ocr.run(&plate);

        string licensePlate=plate.str();
        cout << "================================================\n";
        cout << "License plate number: "<< licensePlate << "\n";
        cout << "================================================\n";
        rectangle(input_image, plate.position, Scalar(0,0,200));
        putText(input_image, licensePlate, Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,200),2);
        if(false){
            imshow("Plate Detected seg", plate.plateImg);
            cvWaitKey(0);
        }

    }
        imshow("Plate Detected", input_image);
       for(;;)
       {
       int c;
       c = cvWaitKey(10);
       if( (char) c == 27)
       break;
       }

    return 0;
}
