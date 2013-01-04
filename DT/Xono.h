#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<list>
#include<map>
#include<set>
#include<vector>
#include<cmath>
#include<iomanip>
#include "time.h"
#include<queue>
using namespace std;

const bool OPTION_CHECK = false;/*check if term appears in all categories*/
const double THRESH_IG = 0.05;/*threshold for term information gain*/
const int THRESH_LEN = 4;/*minimum valid dictionary term length*/
const string TRAIN_DT_PATH = ".\\Data\\train.txt";/*training email file, each line is categoryID + Body*/
const string TEST_DT_PATH = ".\\Data\\test.txt";/*testing email file*/
const string TRAIN_FT_PATH = ".\\Features\\train.txt";/*extracted training feature file*/
const string TEST_FT_PATH = ".\\Features\\test.txt";/*testing feature file, for batch testing*/
const string CATEGORY_SET_PATH = ".\\Analysis\\CategorySet.txt";/*statistics of category set*/
const string INFO_GAIN_PATH = ".\\Analysis\\InfoGain.txt";/*output information gain, for debugging only*/
const string DICT_PATH = ".\\Analysis\\Dict.txt";/*dictionary file learned*/
const string IDF_PATH = ".\\Analysis\\IDF.txt";/*IDF file from training set*/
const double PRUNE_TH = 0.25;/*threshold value for pruning trees*/

inline double entropy(double v)
{
	if (v == 0)
		return 0;
	else
		return v*log(v)/log(2);
}