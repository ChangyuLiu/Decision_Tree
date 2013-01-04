// DT.cpp : Defines the entry point for the console application.
//
#include <windows.h>
#include "Xono.h"
using namespace std;

map<int, map<int,double> > FeatMatrix;	/* FeatMatrix[TermID][#line] =  tfidf*/
map<int, int> CatMatrix;				/* CatMatrix[#line] = catID */
queue<struct node *> splitNode;
queue<DWORD_PTR> cpuQueue;
int threadReady = 0;
int cpuCount = 0;
//vector<struct node *> splitNode;

struct node {
	int type;					/* type-0-internal, type-1-leaf */
	int termID;					/* which term to split */
	double thresh;				/* if type=0, split threshold */
	int catID;					/* if type=1, then output */
	vector<int> *index_ptr;		/* training sample index */
	struct node *left;
	struct node *right;
								/* parent node? later for pruning*/
};

void initialize()
{
	//ifstream initif(TRAIN_FT_PATH.c_str());
	ifstream initif(TEST_FT_PATH.c_str());///////////////////for large training file, it's very slow, need to optimize for each node
	if (!initif.is_open()) {
		cerr << "cannot open train feature file" << endl;
		exit(-1);
	}
	int count = 0;
	string line;
	stringstream linestream;
	int catID;
	int termID;
	double val;
	while(getline(initif, line)) {
		if(line.empty())
			break;
		linestream << line;
		linestream >> catID;
		CatMatrix[count] = catID;
		while(linestream.good()) {
			linestream >> termID;
			linestream >> val;
			FeatMatrix[termID][count] = val;
		}
		count++;
		linestream.clear();
	}
	initif.close();
}

void initroot(struct node *root)
{
	root->type = 0;
	root->thresh = 0;
	root->catID = 0;
	root->termID = 0;
	root->index_ptr = new vector<int>;
	for (int i=0;i!=CatMatrix.size();i++)
		(*(root->index_ptr)).push_back(i);
	root->left = NULL;
	root->right = NULL;
}

double mapEnt(map<int,int> *m)
{
	double sum=0;
	for (map<int,int>::iterator iter=(*m).begin();iter!=(*m).end();iter++)//calculate total num of emails, m[categoryID] = #of emails in this category
		sum += (double)(*iter).second;
	vector<double> ar;
	for (map<int,int>::iterator iter=(*m).begin();iter!=(*m).end();iter++)//
		ar.push_back(((double)(*iter).second)/sum);
	double result = 0;
	for (vector<double>::iterator iter=ar.begin();iter!=ar.end();iter++)
		result += -entropy(*iter);
	return result;
}

double cc(map<int,int> *m)
{
	int result = 0;
	for (map<int,int>::iterator iter=(*m).begin();iter!=(*m).end();iter++)
		result += (*iter).second;
	return (double)result;
}

void split(struct node *p)
{
	/* check if all output have same values */
	int flag_same_out = 1;
	vector<int>::iterator check_ptr=(*(p->index_ptr)).begin();
	int check_cat = CatMatrix[*check_ptr];
	check_ptr++;
	while (check_ptr!=(*(p->index_ptr)).end()) {
		if (check_cat!=CatMatrix[*check_ptr]) {
			flag_same_out = 0;
			break;
		}
		check_ptr++;
	}
	if (flag_same_out) {
		p->type = 1;
		p->termID = -1;
		p->thresh = -1;
		p->catID = check_cat;
		p->index_ptr = NULL;
		p->left = NULL;
		p->right = NULL;
		cout << "not regular return" << endl;
		return;
	}

	/* check if all input are the same values */
	int flag_same_in = 1;
	for (map<int, map<int, double> >::iterator iter=FeatMatrix.begin();iter!=FeatMatrix.end();iter++) {
		if (!flag_same_in)
			break;
		vector<int>::iterator iter1=(*(p->index_ptr)).begin();
		int tempid;
		if ((*iter).second.find(*iter1)==(*iter).second.end())
			tempid = 0;
		else
			tempid = *iter1;
		iter1++;
		while (iter1!=(*(p->index_ptr)).end()) {
			if (tempid!=(*iter1))
				flag_same_in = 0;
			iter1++;
		}
	}
	if (flag_same_in) {
		map<int,int> siopt;
		for (vector<int>::iterator iter=(*(p->index_ptr)).begin();iter!=(*(p->index_ptr)).end();iter++) {
			siopt[CatMatrix[*iter]]++;
		}
		int cnt = -1;
		int opt = -1;
		for (map<int,int>::iterator iter=siopt.begin();iter!=siopt.end();iter++) {
			if ((*iter).second > cnt) {
				cnt = (*iter).second;
				opt = (*iter).first;
			}
		}
		p->type = 1;
		p->termID = -1;
		p->thresh = -1;
		p->catID = opt;
		p->index_ptr = NULL;
		p->left = NULL;
		p->right = NULL;
		cout << "not regular return" << endl;
		return;
	}

	/* inner nodes, do split */
	/* test each term */
	double IGR = -1;
	int split_termID = -1;
	double split_value = -1;
	for (map<int, map<int, double> >::iterator iter=FeatMatrix.begin();iter!=FeatMatrix.end();iter++) {
		set<double> vset;
		set<double> tset;	/* testing threshold, mid point (C4.5 uses left point) */
		for (vector<int>::iterator iter1=(*(p->index_ptr)).begin();iter1!=(*(p->index_ptr)).end();iter1++) {
			if ((*iter).second.find(*iter1)!=(*iter).second.end())
				vset.insert((*iter).second[(*iter1)]);
			else
				vset.insert(0);
		}

		set<double>::iterator a = vset.begin();
		a++;
		set<double>::iterator b = vset.begin();
		while(a!=vset.end()) {
			tset.insert(((*a)+(*b))/2);///////////////use left point or midpoint
			a++;
			b++;
		}
		
		/* for each term, test threshold values in tset */
		for (set<double>::iterator iter1=tset.begin();iter1!=tset.end();iter1++) {
			map<int, int> A;//A[catID] = count
			map<int, int> L;
			map<int, int> R;
			for (vector<int>::iterator iter2=(*(p->index_ptr)).begin();iter2!=(*(p->index_ptr)).end();iter2++) {
				A[CatMatrix[*iter2]]++;
				double temp = 0;
				if ((*iter).second.find(*iter2) == (*iter).second.end())
					temp = 0;
				else
					temp = (*iter).second[*iter2];
				if (temp <= *iter1)
					L[CatMatrix[*iter2]]++;				//left
				else
					R[CatMatrix[*iter2]]++;				//right
			}
			double p_H_up = mapEnt(&A);
			double p_H_down = cc(&L)/(cc(&L)+cc(&R))*mapEnt(&L) + cc(&R)/(cc(&L)+cc(&R))*mapEnt(&R);
			double p_IG = p_H_up-p_H_down;
			double p_IGR = p_IG/p_H_up;////////////use InfoGain or InfoGainRate
			//double p_IGR = p_IG;////////////use InfoGain or InfoGainRate
	//		std::cout << "current term: " << (*iter).first << " InfoGain: " << p_IG << endl;///////////////
			if (p_IGR>IGR) {
				IGR = p_IGR;
				split_termID = (*iter).first;
				split_value = *iter1;
			}
		}
	//	system("PAUSE");//////////////////
	}
	p->type = 0;
	p->termID = split_termID;
	p->thresh = split_value;

	p->left = new struct node;
	p->right = new struct node;
	p->left->index_ptr = new vector<int>;
	p->right->index_ptr = new vector<int>;
	for (vector<int>::iterator iter=(*(p->index_ptr)).begin();iter!=(*(p->index_ptr)).end();iter++) {
		double tmp;
		if (FeatMatrix[p->termID].find(*iter)==FeatMatrix[p->termID].end())
			tmp = 0;
		else
			tmp = FeatMatrix[p->termID][*iter];
		if (tmp <= p->thresh)
			(*(p->left->index_ptr)).push_back(*iter);
		else
			(*(p->right->index_ptr)).push_back(*iter);
	}
//	split(p->left);				//recursion
//	split(p->right);			//recursion
	splitNode.push(p->left);
	splitNode.push(p->right);

	std::cout << "termID: " << p->termID << " thresh: " << p->thresh << endl;/////////////
}


void pntemp(int n)
{
	for (int i=0;i!=n;i++)
		cout << "\t";
}
vector<string> dict;//////////////////
void intepreter(struct node *p,int n)
{
	if (p->type == 1) {
		pntemp(n);
		cout << "Decide category: " << p->catID << endl;
	}
	else {
		pntemp(n);
		//cout << "If term[" << p->termID << "] <= " << p->thresh << endl;
		cout << "If " << dict[p->termID] << " <= " << p->thresh << endl;
		intepreter(p->left,n+1);
		pntemp(n);
		cout << "Else " << endl;
		intepreter(p->right,n+1);
	}
}

void prune(struct node *root)
{
	/* I think for real valued split, no need to prune (prune will do nothing on the tree) */

}

int predict(struct node *p, map<int,double> *feat)
{
	if (p->type == 1)
		return p-> catID;
	else {
		double test_term_val;
		if ((*feat).find(p->termID) == (*feat).end())
			test_term_val = 0;
		else
			test_term_val = (*feat)[p->termID];
		if (test_term_val <= p->thresh)
			return predict(p->left, feat);
		else
			return predict(p->right, feat);
	}
}

double test(struct node *root)
{
	ifstream pifs(TEST_FT_PATH.c_str());
	if (!pifs.is_open()) {
		cerr << "Cannot open testing file!" << endl;
		exit(-1);
	}
	string line;
	stringstream linestream;
	int catID;
	int pcatID;
	int termID;
	double val;
	double countall = 0;
	double counttrue = 0;
	map<int,double> feat;
	while (getline(pifs, line)) {
		if(line.empty())
			break;
		countall++;
		linestream << line;
		linestream >> catID;
		while (!linestream.eof()) {
			linestream >> termID;
			linestream >> val;
			feat[termID] = val;
		}
		pcatID = predict(root, &feat);
		cout << "Actural: " << catID << " Predict: " << pcatID << endl;
		if (pcatID == catID)
			counttrue++;
		feat.clear();
		linestream.clear();
	}
	cout << "Accuracy: " << counttrue/countall << endl;
}

DWORD_PTR WINAPI Func1(void* p){
/*	struct node *root = new struct node;	
	cout << "After root " << endl;
	initialize();
	cout << "initialize " << endl;
	initroot(root);
	cout << "After initroot " << endl;*/
//	cout << "I am in Func1" << endl;
//	cout << "GetCurrentProcessorNumber is " << GetCurrentProcessorNumber() << endl;
	if(splitNode.empty()){
		threadReady++;
//		cout << "Queue is empty" << endl;
		return 0;
	}

	struct node *root = splitNode.front();
	splitNode.pop();
	split(root);
//	cout << "After split " << endl;

//	cout << "I am in Func1" << endl;

	threadReady++;
	cpuQueue.push((DWORD_PTR)GetCurrentProcessorNumber());

	return 0;
}

DWORD_PTR GetNumCPUs() {
  SYSTEM_INFO m_si = {0, };
  GetSystemInfo(&m_si);
  return (DWORD_PTR)m_si.dwNumberOfProcessors;
}
int a;
DWORD_PTR WINAPI threadMain(void* p) {
//	cout << "GetCurrentProcessorNumber is " << GetCurrentProcessorNumber() << endl;
	int count = 1000000000;
	while(count > 0){
		  a = 0;
		//  count = count - 1;
	}

	cout << "I am done" << endl;
	threadReady ++;
	return 0;
}


int main()
{
/*	ifstream ifdict(DICT_PATH.c_str());//////////////
	string line;////////////////
	while (ifdict>>line)////////////////
		dict.push_back(line);/////////////////
	struct node *root = new struct node;		
	initialize();
	initroot(root);
	splitNode.push(root);
	int t1;
//	split(root);
	DWORD_PTR c = GetNumCPUs();
	HANDLE* m_threads = new HANDLE[c];
	while(!splitNode.empty()){
		for(DWORD_PTR i = 0; i < c; i++) {
			DWORD_PTR m_mask = 1 << i;
			m_threads[i] =CreateThread(NULL, 0, Func1, (LPVOID)i, CREATE_SUSPENDED, NULL);
			SetThreadAffinityMask(m_threads[i], m_mask);
			ResumeThread(m_threads[i]);
	//		wprintf(L"Creating Thread %d (0x%08x) Assigning to CPU 0x%08x\r\n", i, (LONG_PTR)m_threads[i], m_mask);
		}

		while(threadReady < 4)
		  Sleep(10);
		threadReady = 0;
	}
	
	t1 = clock();
	cout << "TIME ELAPSED[SPLIT]: " << double(t1)/CLOCKS_PER_SEC << " SECONDS!" << endl;
//	CloseHandle(hThread1);
//	Sleep(400000);
//	system("PAUSE");
//	prune(root);
//	intepreter(root,0);
//	test(root);
	return 0;*/

	ifstream ifdict(DICT_PATH.c_str());//////////////
	string line;////////////////
	while (ifdict>>line)////////////////
		dict.push_back(line);/////////////////
	struct node *root = new struct node;		
	initialize();
	initroot(root);
	splitNode.push(root);
//	split(root);

	int round = 10;

	

	DWORD_PTR c = GetNumCPUs();
	HANDLE *m_threads = NULL;
	m_threads = new HANDLE[c];

	for(DWORD_PTR i = 0; i < c; i++) {
		cpuQueue.push(i);
	}
	while(true){
		if(cpuQueue.size() == 4 && splitNode.empty())
			break;
		if(cpuQueue.size() > 0 && !splitNode.empty()){
			DWORD_PTR m_id = 0;
			DWORD_PTR m_mask = 1 << cpuQueue.front();
			m_threads[cpuQueue.front()] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Func1, (LPVOID)cpuQueue.front(), NULL, &m_id);
			SetThreadAffinityMask(m_threads[cpuQueue.front()], m_mask);
			cpuQueue.pop();
//			cpuCount++;
		}
		Sleep(10);
	}
/*	while(!splitNode.empty()){
		cout << "New round " << endl;
		DWORD_PTR c = GetNumCPUs();
		HANDLE *m_threads = NULL;
		m_threads = new HANDLE[c];
		for(DWORD_PTR i = 0; i < c; i++) {
		
			DWORD_PTR m_id = 0;
			DWORD_PTR m_mask = 1 << i;
			m_threads[i] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Func1, (LPVOID)i, NULL, &m_id);
			SetThreadAffinityMask(m_threads[i], m_mask);
		//	wprintf(L"Creating Thread %d (0x%08x) Assigning to CPU 0x%08x\r\n", i, (LONG_PTR)m_threads[i], m_mask);
		}
	
		while(threadReady < c)
			Sleep(10);
		cout << "threadReady is " << threadReady << endl;
		threadReady = 0;
		cout << endl;
	//	round --;
	}*/
	int t1 = clock();
	cout << "TIME ELAPSED[SPLIT]: " << double(t1)/CLOCKS_PER_SEC << " SECONDS!" << endl;
	return 0;
}



