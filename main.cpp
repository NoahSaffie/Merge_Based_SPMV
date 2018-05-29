/*
  Supply the program with a single .mtx file through command line
  or you can supply it a .txt file that has the following:
  path to directory with .mtx fiiles as first line
  every subsequent line is a filename for an mtx.
  If file names in .txt already include .mtx extension then change REQUIRES_MTX_EXTENSION below to 0
 */
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
extern "C"{
#include "mmio.h"
}

using namespace std;
#define VALUE_TYPE double
#define REQUIRES_MTX_EXTENSION 1
//Credit to: https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()
struct Coord{int x;int y;};

inline void DetermineCoordinate(int diagonal, vector<int>csrRowPtrA, int sizeB, Coord &point);
inline void MergePath(vector<int> csrRowPtrA, vector<int> csrColIndex, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y, double &MergeTime, double &DiagonalTime);
inline void StandardSpMV(vector<int> csrRowPtrA, vector<int> csrColIndexA, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y_verified);
inline void CompareVectors(vector<double> y1, vector<double> y2);
inline int  GetMatrix(string filename, vector<int> &csrRowPtrA, vector<int> &csrColIndexA, vector<VALUE_TYPE> &csrValueA);
inline void RunTests(string filename);
int main(int argc, char* argv[])
{
  /*
    Looks for a Command Line Arguement that is either a single .mtx file
    or a .txt that contains a list of .mtx files, with one per line and nothing else
   */
  
  ofstream outputFile;
  outputFile.open("output.csv", ios::out);
  outputFile << "Matrix,Non-Zeros,Average time to find Coordinate along diagonal,Avergae Path Traversal and Matrix Multipication,Average total time for MergePath Function" << endl; //Header
  outputFile.close();
  string filename;
  if(argc < 3)
    {
      filename = argv[1];
      cout << "Recieved a file name of: " << filename << endl;
    }
  else
    {
      cout << "Failed to recieve or recongize, a command line argument for a file of .mtx" << endl;
    }
  if(filename.find(".txt") != string::npos)
    {
      string path;
      string nextFile;
      fstream inputFiles;
      inputFiles.open(filename);
      inputFiles >> path;
      while(inputFiles.peek() != EOF)
	{
	  inputFiles >> nextFile;
	  string fileWithPath(path);
	  fileWithPath.append(nextFile);
	  if(REQUIRES_MTX_EXTENSION){	  fileWithPath.append(".mtx");}
	  cout << "Processing file: " << fileWithPath << endl;
	  RunTests(fileWithPath);
	}
    }
  else if(filename.find(".mtx") != string::npos)
    {
	RunTests(filename);
    }
  else{cout << "Failed to receive or recongize a .mtx or .txt" << endl;}
  return 0;
}
inline void DetermineCoordinate(int diagonal, vector<int> csrRowPtrA, int sizeB, Coord &point)
{
  int sizeA = csrRowPtrA.size()-1; //-1 to fix to the vector we want (ignore first element)
  int x_min = max(diagonal - sizeB, 0); //First option because a diagonal can go "off"/below the table and in that final case we can find where it first comes on to the table to be the minimum
  int x_max = min(diagonal, sizeA); //First option for the very early cases where the diagonal ends before the end of RowPtr
  int x_mid;
  while(x_min != x_max)
    {
      x_mid = (x_min+x_max) >> 1;
      if((diagonal-x_mid) > csrRowPtrA.at(x_mid+1))
	{
	  x_min = x_mid;
	}
      else
	{
	  x_max = x_mid;
	}
    }
  point.x = min(x_min, sizeA-1); //Double check it is valid index
  point.y = diagonal-x_min;
}
inline void MergePath(vector<int> csrRowPtrA, vector<int> csrColIndex, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y, double &MergeTime, double &DiagonalTime)
{
  /*
    Changes: 
    Abandoned ListB entirely, can accomplish its use by simply having a 'y' value when finding coordinates 
    Simplified the section about a row spanning across more than one team, no needed overhead like before
    Calculate IPT slightly different (In a more base logical way [Possible I just miss why the other way is better])
    Simpler basis for determing diagonals - Works entirely based the Items per thread
   */
  /*
    Drawbacks/Considerations:
    Checking if OpenMP is actually enabled and not operating under the assumption it is
   */
  /*
    Notes:
    startCoord.y  is the value(an index) of listB
    IPT - Items Per Thread
    Getting totalData - x[csrColIndex[startCoord.y]] is a part of our selective dot product (Ignoring 0 sums)
        We are doing row*column, we get row value from:  Value<row, column> where row is set by outer loop, and column is startCood.y (this is as if in Matrix)
        For the column we want the value in X that corresponds to the column  in the Matrix we are working in so x[csrColIndex[startCoord.y]] does this
    totalPathLength check is because it is the final y value for the diagonal to be, that it will intersect perfectly for the very last coordinate
    it should never be necessary on the startDiagonal but it is not impossible that it be needed so put just in case
    THe -1, will cancel out the +1 if they were easily divisible. The +1 will help fix the unbalanced load that would result on the final team/thread i the result of uneven division when finding IPT  
   */
  
  int num_of_threads = omp_get_num_threads();
#pragma omp parallel for num_threads(num_of_threads) schedule(static) default(none) firstprivate(num_of_threads, x, csrValueA, csrColIndex, csrRowPtrA) shared(y, MergeTime, DiagonalTime)
  for(int i = 0; i < num_of_threads; i++)
    {
      int sizeA = csrRowPtrA.size()-1;
      int sizeB = csrColIndex.size();
      int totalPathLength = sizeA+sizeB;
      int IPT = ((totalPathLength-1)/num_of_threads)+1;
      TimeVar diagonalStart = timeNow();
      
      Coord startCoord;
      int startDiagonal = min(IPT*i, totalPathLength);
      DetermineCoordinate(startDiagonal, csrRowPtrA, sizeB, startCoord);
      
      Coord endCoord;
      int endDiagonal = min(startDiagonal+IPT, totalPathLength);
      DetermineCoordinate(endDiagonal, csrRowPtrA, sizeB, endCoord);
      
      TimeVar diagonalEnd = timeNow();
      #pragma omp critical(TimeOutputDiagonal)
      DiagonalTime += duration(diagonalEnd - diagonalStart);
      TimeVar mergeStart = timeNow();
      int loopY;
      double totalData = 0.0;
      for(; startCoord.x <= endCoord.x; ++startCoord.x, totalData = 0.0)
	{   
	  int maxYforLoop = min(csrRowPtrA.at(startCoord.x+1)-1, endCoord.y);
	  #pragma omp simd
	  for(loopY = startCoord.y; loopY <= maxYforLoop; ++loopY)
	    {
	      totalData += (csrValueA.at(loopY) * x.at(csrColIndex.at(loopY)));
	    }
	  startCoord.y = loopY;
          #pragma omp critical(OutputUpdate)
	  y.at(startCoord.x) = y.at(startCoord.x) +  totalData;
	}
      TimeVar mergeEnd = timeNow();
      #pragma omp critical(TimeOutputMerge)
      MergeTime += duration(mergeEnd-mergeStart);
    }
  #pragma omp barrier
}
inline void RunTests(string filename)
{
  //int TESTS_FOR_SINGLE_FILE = testToRunOnFile;
  //Vector Declarations/Inits
  vector<int> RowPtrA;
  vector<int> ColIndexA;
  vector<VALUE_TYPE> ValueA;

  //File I/O
  ofstream outputFile;
  outputFile.open("output.csv", ios::out | ios::app);
  string separator(","); //comma for CSV
  //outputFile << "Matrix" << separator << "Non-zeros" << separator << "Time for Diagonal fucntion" << separator << "Time for Merge path Loop" << separator << "Time for total Merge Function" << endl;
  //Start processing matrix - Base results and CSR only need to be found once.
  int columns = GetMatrix(filename, RowPtrA, ColIndexA, ValueA);
  if(columns < 0)
    {
      outputFile.close();
      return;
    }
  int nnzA = ColIndexA.size();
  int rowsA = RowPtrA.size();
  int TESTS_FOR_SINGLE_FILE = max(5, 16000000/nnzA);
  vector<double> x(columns, 1.0);
  vector<double> y(rowsA, 0.0);
  vector<double> y_verified(rowsA, 0.0);
  //Might want to loop this a couple of times for a more accurate average
  //double baseTotal = 0.0;
  //TimeVar baseTime = timeNow();
  StandardSpMV(RowPtrA, ColIndexA, ValueA, x, y_verified);
  //TimeVar baseEndTime = timeNow();
  //baseTotal+=duration(baseEndTime-baseTime);
  double funcTotal = 0;
  double mergeTotal = 0;
  double diagonalTotal = 0;
  //Run the Mergebased Spmv function multiple times to find a more accurate average time value
  for(int i = 0; i < TESTS_FOR_SINGLE_FILE; i++)
    {
      double singleMergeTime = 0.0;
      double singleDiagonalTime = 0.0;
      fill(y.begin(), y.end(), 0.0);
      
      TimeVar singleFuncStart = timeNow();
      MergePath(RowPtrA, ColIndexA, ValueA, x, y, singleMergeTime, singleDiagonalTime);
      TimeVar singleFuncEnd = timeNow();
      
      double singleFuncTime = duration(singleFuncEnd-singleFuncStart);
      CompareVectors(y, y_verified);
      funcTotal+=singleFuncTime;
      mergeTotal+=singleMergeTime;
      diagonalTotal+=singleDiagonalTime;
    }
  //Change filename to only include file not path
  outputFile << filename.substr(filename.find_last_of("/\\")+1) << separator << nnzA << separator << diagonalTotal/TESTS_FOR_SINGLE_FILE << separator << mergeTotal/TESTS_FOR_SINGLE_FILE << separator << funcTotal/TESTS_FOR_SINGLE_FILE << endl;
  outputFile.close();
}
inline int GetMatrix(string filename, vector<int> &csrRowPtrA, vector<int> &csrColIndexA, vector<VALUE_TYPE> &csrValueA)
{  
  int columnsA, nnzA_mtx, nnzA, rowsA;
  //Based on example by MM
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  f = fopen(filename.c_str(), "r");
  int isInteger = 0, isReal = 0, isSymmetric = 0, isBinary = 0;
  if (f == NULL)
    return -1;

  if ( mm_read_banner(f, &matcode) != 0)
    {
      cout << "Could not process Matrix Market banner." << endl;
      return -2;
    }
  //cout << "Matcode: " << matcode << endl;
  //cout << mm_typecode_to_str(matcode) << endl;
  if( mm_is_pattern( matcode) )
    {
      isBinary = 1;
    }
  if ( mm_is_complex( matcode ) )
    {
      cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
      return -3;
    }
  if ( mm_is_real ( matcode) )     { isReal = 1; }
  if ( mm_is_integer ( matcode ) ) { isInteger = 1; }
  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &rowsA, &columnsA, &nnzA_mtx)) !=0)
    {
      return -4;
    }
  if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
      isSymmetric = 1;
    }
  
  vector<int> temp_csrRowPtrA(rowsA+1, 0);
  vector<int> temp_csrRowIndexA(nnzA_mtx); 
  vector<int> temp_csrColIndexA(nnzA_mtx);
  vector<VALUE_TYPE> temp_csrValueA(nnzA_mtx);

#pragma omp parallel for default(none) firstprivate(f, nnzA_mtx, isInteger, isReal, isBinary) shared(temp_csrRowPtrA, temp_csrRowIndexA, temp_csrColIndexA, temp_csrValueA)
  for (int k = 0; k < nnzA_mtx; k++)
    {
      int i =-1, j = -1;
      VALUE_TYPE double_value;
      int int_value;
      //Can only scan in safely one at a time, otherwise might break future scans
      #pragma omp critical(InputScan)
      {
      //If doubles
      if (isReal)
	fscanf(f, "%d %d %lg\n", &i, &j, &double_value);
      //If Ints
      else if (isInteger)
        {
	  fscanf(f, "%d %d %d\n", &i, &j, &int_value);
        }
      else if(isBinary)
	{
	  fscanf(f, "%d %d\n", &i, &j);
	}
      }
      // adjust from 1-based to 0-based
      i--;
      j--;

      //Add the value/indexes to respective arrays
      temp_csrRowPtrA.at(i)++;
      temp_csrRowIndexA.at(k) = i;
      temp_csrColIndexA.at(k) = j;
      if(isReal)
	{
	  temp_csrValueA.at(k) = double_value;
	}
      else if(isInteger)
	{
	  temp_csrValueA.at(k) = int_value*1.0;
	}
      else if(isBinary)
	{
	  temp_csrValueA.at(k) = 1.0;
	}
    }
  fclose(f);
  
  if (isSymmetric)
    {
      #pragma omp parallel for default(none) firstprivate(nnzA_mtx) shared(temp_csrRowIndexA, temp_csrRowPtrA, temp_csrColIndexA)
      for (int i = 0; i < nnzA_mtx; i++)
	{
	  if (temp_csrRowIndexA.at(i) != temp_csrColIndexA.at(i))
	    temp_csrRowPtrA.at(temp_csrColIndexA.at(i))++;
	}
    }
  // exclusive scan for temp_csrRowPtrA
  int old_val, new_val;
  //Turns into a correct rowptr array (index of valueArray where a new row begins)
  //Ex. Would make [4,2,0,6,7] -> [0,4,6,6,12]
  //no openMP
  old_val = temp_csrRowPtrA[0];
  temp_csrRowPtrA[0] = 0;
  for (int i = 1; i <= rowsA; i++)
    {
      new_val = temp_csrRowPtrA.at(i);
      temp_csrRowPtrA.at(i) = old_val + temp_csrRowPtrA.at(i-1);
      old_val = new_val;
    }
  csrRowPtrA = temp_csrRowPtrA;
  //Will enlarge if we are doing a symmetric matrix
  nnzA = temp_csrRowPtrA.at(rowsA);
  
  fill(temp_csrRowPtrA.begin(), temp_csrRowPtrA.end(), 0);
  //this will order the (potentially) unordered data that is currently in the temp arrays
  //however it doesn't completely order it, columnIndexes/Value Indexes will not be in exact order
  //But it shouldn't matter everything is in correct row sections and the column index, and value index are still a valid pair
  csrColIndexA = vector<int>(nnzA);
  csrValueA = vector<VALUE_TYPE>(nnzA);
  #pragma omp parallel for default(none) shared(csrColIndexA, csrValueA, temp_csrRowPtrA) firstprivate(temp_csrRowIndexA, temp_csrColIndexA, temp_csrValueA, isSymmetric, csrRowPtrA, nnzA_mtx)
  for (int i = 0; i < nnzA_mtx; i++)
    {
      int tempRowIndexAti = temp_csrRowIndexA.at(i);
      int offset = csrRowPtrA.at(tempRowIndexAti) + temp_csrRowPtrA.at(tempRowIndexAti)++;
      csrColIndexA.at(offset) = temp_csrColIndexA.at(i);
      csrValueA.at(offset) = temp_csrValueA.at(i);
      //Not on the diagonal (like y = -x) that the symmetry is defined around
      if (isSymmetric && tempRowIndexAti != temp_csrColIndexA.at(i))
	{
	  offset = csrRowPtrA.at(temp_csrColIndexA.at(i)) + temp_csrRowPtrA.at(temp_csrColIndexA.at(i))++;
	  csrColIndexA.at(offset) = temp_csrRowIndexA.at(i);
	  csrValueA.at(offset) = temp_csrValueA.at(i);
	}
    }
  return columnsA;
}
inline void StandardSpMV(vector<int> csrRowPtrA, vector<int> csrColIndexA, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y_verified)
{
  int sizeA = csrRowPtrA.size();
  double totalData = 0.0;
  for(int i = 0; i < sizeA-1; i++, totalData = 0.0)
    {
      for(int j = csrRowPtrA.at(i); j <  csrRowPtrA.at(i+1); j++)
	{
	  totalData += (csrValueA.at(j) * x.at(csrColIndexA.at(j)));
	}
      y_verified.at(i) = totalData;
    }
}
inline void CompareVectors(vector<double> y1, vector<double> y2)
{
  if(y1 == y2)
    {
      //cout << "Success!" << endl;
    }
  else
    {
      cout << "Failure." <<  endl;
      for(unsigned i = 0; i < y1.size(); i++)
	{
	  cout << "At line: " << i << " of size " << y1.size() << endl;
	  if(y1.at(i) != y2.at(i))
	    {
	      cout << "\ty1: " << y1.at(i) << "\ty2: " << y2.at(i) << endl;
	    }
	}
    }
}
