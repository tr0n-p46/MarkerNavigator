#include "stdafx.h"
#include "tserial.h"

#include <queue>
#include <stack>
#include <fstream>
#include <iostream>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <tesseract\baseapi.h>

#define DEFAULT_CAPTURE_INDEX 0
#define EXTERNAL_CAPTURE_INDEX 1

using namespace cv;
using namespace std;

int captureIndex;
VideoCapture capture;

bool FLAG_MAXIMIZE = true;

Point2f endzone;


class CharacterList
{
public: char c;
        Point2f centroid;
        Point2f getCentroid()
        {
            return centroid;
        }
};

string resultExpression;
double value;



queue<CharacterList> expression;

vector<CharacterList> operands;
vector<CharacterList> operators;
vector<CharacterList> braces;

double thresholdDistance = 100.0;
double thresholdAngle = 25.0;

Tserial *com = new Tserial();

class ThresholdValue
{
public: int HSV[2][3];
        Scalar getLow()
        {
            return Scalar(HSV[0][0], HSV[0][1], HSV[0][2]);
        }
        Scalar getHigh()
        {
            return Scalar(HSV[1][0], HSV[1][1], HSV[1][2]);
        }
};

ThresholdValue endZone;
ThresholdValue characters;
ThresholdValue frontMarker;
ThresholdValue backMarker;

void saveThresholdData(string name)
{
    ofstream fThreshold;
    fThreshold.open("Threshold.txt", ios_base::app);
    if (!fThreshold.is_open())
    {
        cout << "Could not open file." << endl;
        exit(0);
    }

    namedWindow(name, CV_WINDOW_AUTOSIZE);

    int iLowH = 0;
    int iHighH = 179;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;

    createTrackbar("LowH", name, &iLowH, 179);
    createTrackbar("HighH", name, &iHighH, 179);

    createTrackbar("LowS", name, &iLowS, 255);
    createTrackbar("HighS", name, &iHighS, 255);

    createTrackbar("LowV", name, &iLowV, 255);
    createTrackbar("HighV", name, &iHighV, 255);
   
    while (true)
    {
        Mat BGRFrame;// = imread("D:\\test.jpg", CV_LOAD_IMAGE_COLOR);
        bool bSuccess = capture.read(BGRFrame);

        if (!bSuccess)
        {
            cout << "Cannot read a frame from video stream" << endl;
            exit(0);
        }

        Mat HSVFrame;
        cvtColor(BGRFrame, HSVFrame, COLOR_BGR2HSV);

        Mat thresholdFrame;
        inRange(HSVFrame, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), thresholdFrame);


        erode(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        dilate(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        dilate(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        erode(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        imshow("Thresholded Result", thresholdFrame);
        if (waitKey(30) == 13)
        {
            fThreshold << iLowH << " " << iHighH << " " << iLowS << " " << iHighS << " " << iLowV << " " << iHighV << endl;
            break;
        }
    }
    destroyAllWindows();
    fThreshold.close();
}

void loadThresholdData()
{
    ifstream fThreshold;
    fThreshold.open("Threshold.txt", ios_base::in);
    if (!fThreshold.is_open())
    {
        cout << "Could not read file.\n";
        exit(0);
    }

    for (int j = 0; j < 3; j++)
    {
        fThreshold >> endZone.HSV[0][j];
        fThreshold >> endZone.HSV[1][j];
    }

    for (int j = 0; j < 3; j++)
    {
        fThreshold >> characters.HSV[0][j];
        fThreshold >> characters.HSV[1][j];
    }

    for (int j = 0; j < 3; j++)
    {
        fThreshold >> frontMarker.HSV[0][j];
        fThreshold >> frontMarker.HSV[1][j];
    }

    for (int j = 0; j < 3; j++)
    {
        fThreshold >> backMarker.HSV[0][j];
        fThreshold >> backMarker.HSV[1][j];
    }
    fThreshold.close();
}

void displayCharacters()
{
    cout << "Operands : \n";
    for (int i = 0; i < operands.size(); i++)
    {
        cout << operands.at(i).c << " -- > " << operands.at(i).centroid << endl;
    }
    cout << "Operators : \n";
    for (int i = 0; i < operators.size(); i++)
    {
        cout << operators.at(i).c << " -- > " << operators.at(i).centroid << endl;
    }
    cout << "Braces : \n";
    for (int i = 0; i < braces.size(); i++)
    {
        cout << braces.at(i).c << " -- > " << braces.at(i).centroid << endl;
    }
}


void initialize()
{
    cout << "Welcome to sheldon\nSelect capture interface\n1. Default internal camera \n2. External camera\nChoice : ";
    int choice;
    cin >> choice;
    while (choice != 1 && choice != 2)
    {
        cout << "Invalid choice!\n1. Default camera \n2. External camera\nChoice : ";
        cin >> choice;
    }
    if (choice == 1)
    {
        cout << "Default camera selected." << endl;
        captureIndex = DEFAULT_CAPTURE_INDEX;
    }
    else
    {
        cout << "External camera selected." << endl;
        captureIndex = EXTERNAL_CAPTURE_INDEX;
    }
    capture = VideoCapture(captureIndex);
    if (!capture.isOpened())
    {
        cout << "Cannot open the capture interface." << endl;
        exit(0);
    }
    cout << "Thresholding : \n1. New threshold.\n2. Load threshold data.\nChoice : " << endl;
    cin >> choice;
    while (choice != 1 && choice != 2)
    {
        cout << "Invalid choice!\n1. Threshold now\n2. Skip thresholding\nChoice : " << endl;
        cin >> choice;
    }
    if(choice == 1)
    {
        remove("Threshold.txt");
        cout << "\nPress enter to save data while thresholding in progress.\n";
        cout << "Press any key to threshold end zone : ";
        cin >> choice;
        saveThresholdData("End Zone");

        cout << "End zone threshold completed.\n\nPress any key to threshold characters : ";
        cin >> choice;
        saveThresholdData("Characters");
       
        cout << "Characters threshold completed.\n\nPress any key to threshold front marker : ";
        cin >> choice;
        saveThresholdData("Front Marker");

        cout << "Front marker thresholded.\n\nPress any key to threshold back marker : ";
        cin >> choice;
        saveThresholdData("Back Marker");

        cout << "Thresholding done.\n";
    }

    loadThresholdData();
    cout << "Threshold data loaded successfully.\n";
}

char getValidChar(string text)
{
    for (int i = 0; i < text.length(); i++)
    {
        char c = text[i];
        if (isdigit(c))
            return c;
        if (c == 'X' || c == 'x')
            return '*';
        switch (c)
        {
        case '+':
        case '-':
        case '/':
        case '(':
        case ')':
        case '{':
        case '}':
        case '[':
        case ']':
        case '^':
            return c;
        }
    }
    return 'a';
}

Mat getThreshold(ThresholdValue color)
{
    Mat BGRFrame;// = imread("D:\\test.jpg", CV_LOAD_IMAGE_COLOR);
    bool bSuccess = capture.read(BGRFrame);

    if (!bSuccess)
    {
        cout << "Cannot read a frame from video stream" << endl;
        exit(0);
    }
    Mat HSVFrame;
    cvtColor(BGRFrame, HSVFrame, COLOR_BGR2HSV);
    Mat thresholdFrame;

    inRange(HSVFrame, color.getLow(), color.getHigh(), thresholdFrame);

    erode(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    dilate(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));


    dilate(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    erode(thresholdFrame, thresholdFrame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    return thresholdFrame;
}
Point2f getCentroid(ThresholdValue color)
{
    Mat thresholdFrame = getThreshold(color);
    IplImage ipl_img = thresholdFrame;
    IplImage* img = &ipl_img;
    CvSeq* contours;
    CvMemStorage *storage = cvCreateMemStorage(0);
    cvFindContours(img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
    double maxArea = 0.0;
    CvSeq* maxContour = contours;
    while (contours)
    {
        if (cvContourArea(contours) > maxArea)
        {
            maxArea = cvContourArea(contours);
            maxContour = contours;
        }
        contours = contours->h_next;
    }
    CvMoments m;
    CvMoments* mu = &m;
    cvMoments(maxContour, mu, false);
    Point2f mc(mu->m10 / mu->m00, mu->m01 / mu->m00);
    return mc;
}

bool isOperator(char c)
{
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '!')
        return true;
    return false;
}

bool isBrace(char c)
{
    if (c == '(' || c == ')' || c == '{' || c == '}' || c == '[' || c == ']')
        return true;
    return false;
}
void loadArenaData()
{
    endzone = getCentroid(endZone);
    cout << "End : (" << endzone.x << ", " << endzone.y << ")" << endl;

    Mat thresholdChars = getThreshold(characters);
    Mat invertChars = Scalar::all(255) - thresholdChars;

    IplImage ipl_img = thresholdChars;
    IplImage* img = &ipl_img;
    CvSeq* contours;
    CvMemStorage *storage = cvCreateMemStorage(0);

    cvFindContours(img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
    while (contours->h_next)
    {
        contours = contours->h_next;
    }
    while (contours)
    {
        Mat dst = Mat::ones(thresholdChars.rows, thresholdChars.cols, CV_8U);

        IplImage d = dst;
        IplImage* pD = &d;

        cvDrawContours(pD, contours, Scalar(255, 255, 255), Scalar(255, 255, 255), 2, CV_FILLED, 8);

        Mat background = Scalar::all(255) - cvarrToMat(pD);
        Mat character = invertChars - background;

        CvMoments m;
        CvMoments* mu = &m;
        cvMoments(contours, mu, false);
        Point2f mc(mu->m10 / mu->m00, mu->m01 / mu->m00);
        //cout << "(" << mc.x << "," << mc.y << ") = ";

        tesseract::TessBaseAPI api;
        api.Init(NULL, "eng", tesseract::OEM_DEFAULT);
        api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        api.SetImage((uchar*)character.data, character.cols, character.rows, 1, character.cols);
        string out = api.GetUTF8Text();
        char op = getValidChar(out);
        cout << op << endl;

        CharacterList ch;
        ch.c = op;
        ch.centroid = mc;
        if (isdigit(op))
            operands.push_back(ch);
        else if(isOperator(op))
            operators.push_back(ch);
        else if (isBrace(op))
            braces.push_back(ch);

        while (true) {
            imshow("character", character);
            if (waitKey(30) == 27)
                break;
        }
       

        contours = contours->h_prev;
        if(contours != NULL)
            contours->h_next = NULL;
    }

}

double evaluatePostfix(string postfix)
{
    double value = 0.0;
    stack<double> stack;
    for (int i = 0; i < postfix.length(); i++)
    {
        char c = postfix[i];
        double a, b, result;
        if (isdigit(c))
        {
            result = c - '0';
        }
        else
        {
            a = stack.top();
            stack.pop();
            b = stack.top();
            stack.pop();
            switch (c)
            {
            case '+': result = b + a;
                break;
            case '-': result = b - a;
                break;
            case '*': result = b * a;
                break;
            case '/': if (a == 0) return -1.0; else result = b / a;
            }
        }
        stack.push(result);
    }
    return stack.top();
}

int priority(char c)
{
    switch (c)
    {
    case '/': return 4;
        break;
    case '*': return 3;
        break;
    case '+': return 2;
        break;
    case '-': return 1;
    }
}

int weight(char c)
{
    switch (c)
    {
    case '*': return 4;
        break;
    case '+': return 3;
        break;
    case '-': return 2;
        break;
    case '/': return 1;
    }
}

int weightMin(char c)
{
    switch (c)
    {
    case '+': return 4;
        break;
    case '/': return 3;
        break;
    case '-': return 2;
        break;
    case '*': return 1;
    }
}


int weightMin2(char c)
{
    switch (c)
    {
    case '*': return 4;
        break;
    case '/': return 3;
        break;
    case '+': return 2;
    }
}

string infixToPostfix(string infix)
{
    string exp = "";
    stack<char> stack;
    for (int i = 0; i < infix.length(); i++)
    {
        char c = infix[i];
        if (isdigit(c))
            exp += c;
        else
        {
            while (!stack.empty() && priority(c) <= priority(stack.top()))
            {
                exp += stack.top();
                stack.pop();
            }
            stack.push(c);
        }
    }
    while (!stack.empty())
    {
        exp += stack.top();
        stack.pop();
    }
    return exp;
}

void generateExpression()
{
    int p = -1, q = -1;
    for (int i = 0; i < 2 * operands.size() - 1; i++)
    {
        CharacterList temp;
        if (i % 2 == 0)
            temp = operands.at(++p);
        else
            temp = operators.at(++q);
        expression.push(temp);
        resultExpression += temp.c;
    }
    value = evaluatePostfix(infixToPostfix(resultExpression));
}

void orient()
{

}

void move()
{

}

void beep()
{

}
double getDistance(Point2f point1, Point2f point2)
{
    double distance = sqrt(pow(abs(point2.y - point1.y), 2.0) + pow(abs(point2.x - point1.x), 2.0));
    return distance;
}

void sendData(char data)
{
    com->sendChar((char)data);
}

double getBotOrientation(Point2f character)
{
    Point2f front(getCentroid(frontMarker));
    Point2f back(getCentroid(backMarker));

    /*double botAngle = tanh((front.y - back.y) / (front.x - back.x)) * 180.0 / 3.142;
    double charAngle = tanh((character.y - back.y) / (character.x - back.x)) * 180.0 / 3.142;
    double angle = botAngle - charAngle;*/

    double botAngle = atan2(front.y - back.y, front.x - back.x) * 180.0 / 3.142;
    double charAngle = atan2(character.y - back.y, character.x - back.x) * 180.0 / 3.142;
    double angle = botAngle - charAngle;
    return abs(angle);
}

Point2f getBotPosition()
{
    Point2f front = getCentroid(frontMarker);
    Point2f back = getCentroid(backMarker);
    Point2f position((front.x + back.x) / 2.0, (front.y + back.y) / 2.0);
    return position;
}

void gotoCharacter(Point2f charPosition)
{
    Tserial *com = new Tserial();
    com->connect("COM8", 57600, spNONE);

    Point2f botPosition = getBotPosition();
    double distance = getDistance(botPosition, charPosition);
    double botOrientation = getBotOrientation(charPosition);

    while (distance > thresholdDistance)
    {
        while (botOrientation > thresholdAngle)
        {
            if (botOrientation > 180.0)
            {
                com->sendChar((char)'3');
            }
            else if(botOrientation < 180.0)
            {
                com->sendChar((char)'4');
            }
            botOrientation = getBotOrientation(charPosition);
        }
        com->sendChar((char)'1');
        botPosition = getBotPosition();
        distance = getDistance(botPosition, charPosition);
        botOrientation = getBotOrientation(charPosition);
    }
    com->sendChar((char)'5');
    com->sendChar((char)'6');
    com->disconnect();
}

void traverse()
{
    while(!expression.empty())
    {
        gotoCharacter(expression.front().centroid);
        cout << expression.front().c << " -- > " << expression.front().centroid << endl;
        expression.pop();
    }
    gotoCharacter(endzone);
}

void sortAscending(int lb, int ub)
{
    for (int i = 0; i < ub - lb; i++)
    {
        for (int j = lb; j < ub - 1; j++)
        {
            int dig1 = (int)operands.at(j).c - 48;
            int dig2 = (int)operands.at(j + 1).c - 48;

            if (dig1 > dig2)
            {
                CharacterList ch = operands.at(j);
                operands.at(j) = operands.at(j + 1);
                operands.at(j + 1) = ch;
            }
        }
    }
}

void sortDescending(int lb, int ub)
{
    for (int i = 0; i < ub - lb; i++)
    {
        for (int j = lb; j < ub - 1; j++)
        {
            int dig1 = (int)operands.at(j).c - 48;
            int dig2 = (int)operands.at(j + 1).c - 48;
            cout << "Dig1 : " << dig1 << "Dig2 : " << dig2 << endl;
            if (dig1 < dig2)
            {
                CharacterList ch = operands.at(j);
                operands.at(j) = operands.at(j + 1);
                operands.at(j + 1) = ch;
            }
        }
    }
}

void sortOperatorsMin2()
{
    for (int i = 0; i < operators.size(); i++)
    {
        for (int j = 0; j < operators.size() - 1; j++)
        {

            if (weightMin2(operators.at(j).c) < weightMin2(operators.at(j + 1).c))
            {
                CharacterList ch = operators.at(j);
                operators.at(j) = operators.at(j + 1);
                operators.at(j + 1) = ch;
            }
        }
    }
}

void sortOperatorsMin()
{
    for (int i = 0; i < operators.size(); i++)
    {
        for (int j = 0; j < operators.size() - 1; j++)
        {

            if (weightMin(operators.at(j).c) < weightMin(operators.at(j + 1).c))
            {
                CharacterList ch = operators.at(j);
                operators.at(j) = operators.at(j + 1);
                operators.at(j + 1) = ch;
            }
        }
    }
}
void sortOperators()
{
    for (int i = 0; i < operators.size(); i++)
    {
        for (int j = 0; j < operators.size() - 1; j++)
        {

            if (weight(operators.at(j).c) < weight(operators.at(j + 1).c))
            {
                CharacterList ch = operators.at(j);
                operators.at(j) = operators.at(j + 1);
                operators.at(j + 1) = ch;
            }
        }
    }
}

int getIndex(char c)
{
    for (int i = 0; i < operators.size(); i++)
    {
        if (operators.at(i).c == c)
            return i;
    }
    return -1;
}

void round1()
{
    if (FLAG_MAXIMIZE)
    {
        sortDescending(0, operands.size());
        sortOperators();
        int index = getIndex('-');
        if (index == -1)
            index = getIndex('/');
        if (index != -1)
            sortAscending(index + 1, operands.size());
        generateExpression();
    }
    else
    {
        sortAscending(0, operands.size());
        if (getIndex('-') != -1)
        {
            sortOperatorsMin();
        }
        else
        {
            sortOperatorsMin2();
            int index = getIndex('/');
            if (index != -1)
            {
                sortDescending(index + 1, operands.size());
            }
        }
        generateExpression();
    }
}
void round2()
{
    if (FLAG_MAXIMIZE)
    {

    }
    else
    {

    }
}
void round3()
{

}


int main(int argc, char** argv)
{
    initialize();
    int round;
    cout << "Enter round : ";
    cin >> round;
   
    cout << "Enter mode of operation :\n1. Maximize\n2. Minimize\nChoice : ";
    int choice;
    cin >> choice;
    while (choice != 1 && choice != 2)
    {
        cout << "Invalid choice!\n1. Maximize\n2. Minimize\nChoice : ";
        cin >> choice;
    }
    if (choice == 2)
        FLAG_MAXIMIZE = false;
    cout << "Loading arena data..." << endl;
    loadArenaData();
    cout << "Arena loaded successfully." << endl;
    displayCharacters();

    cout << "Evaluating expression...";

   
   
    if (round == 1)
        round1();
    else if (round == 2)
        round2();
    else round3();

    cout << "Expression generated.\n\n";
    if (FLAG_MAXIMIZE)
        cout << "Maximum :\n";
    else
        cout << "Minimum :\n";

    cout << resultExpression << " = " << value << endl;

    cout << "Robot traversal progress...\n";
    traverse();
    cout << "Objective completed! Thank you...\n\n";
    cin >> choice;
    getchar();
    return 0;

}