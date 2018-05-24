#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
extern "C"
{
#include "mmio.h"
}
using namespace std;
#define VALUE_TYPE double

//Credit to: https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()
struct Coord
{
  int x;
  int y;
};

inline void DetermineCoordinate(int diagonal, vector<int>csrRowPtrA, int sizeB, Coord &point);
inline void MergePath(vector<int> csrRowPtrA, vector<int> csrColIndex, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y, double &MergeTime, double &DiagonalTime);
inline void StandardSpMV(vector<int> csrColIndexA, vector<int> csrRowPtrA, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y_verified);
inline void CompareVectors(vector<double> y1, vector<double> y2);
inline void GetMatrix(char* filename, vector<int> &csrRowPtrA, vector<int> &csrColIndexA, vector<VALUE_TYPE> &csrValueA);
inline void RunTests(int argc, char* argv[], int testsToRunOnFile);
int main(int argc, char* argv[])
{
  RunTests(argc, argv, 1000);
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
      x_mid = (x_min+x_max) >> 2;
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
  #pragma omp parallel for num_threads(num_of_threads) default(none) firstprivate(num_of_threads, x, csrValueA, csrColIndex, csrRowPtrA) shared(y, MergeTime, DiagonalTime)
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
      double totalData;
      for(; startCoord.x <= endCoord.x; ++startCoord.x, totalData = 0.0)
	{
	  for(; startCoord.y <= endCoord.y && csrRowPtrA.at(startCoord.x+1) > startCoord.y; ++startCoord.y)
	    {
	      totalData += (csrValueA.at(startCoord.y) * x.at(csrColIndex.at(startCoord.y)));
	    }
          #pragma omp critical(OutputUpdate)
	  y.at(startCoord.x) = y.at(startCoord.x) +  totalData;
	}
      TimeVar mergeEnd = timeNow();
      #pragma omp critical(TimeOutputMerge)
      MergeTime += duration(mergeEnd-mergeStart);
    }
}
inline void RunTests(int argc, char* argv[], int testToRunOnFile)
{
  ofstream outputFile;
  outputFile.open("output.txt", ios::out | ios::app);
  int TESTS_FOR_SINGLE_FILE = testToRunOnFile;
  vector<int> RowPtrA;
  vector<int> ColIndexA;
  vector<VALUE_TYPE> ValueA;
  char* filename;
  if(argc < 3)
    {
      filename = argv[1];
    }
  else
    {
      cout << "Failed to recieve or recongize, a command line argument for a file of .mtx" << endl;
    }
  GetMatrix(filename, RowPtrA, ColIndexA, ValueA);
  int nnzA = ColIndexA.size();
  int rowsA = RowPtrA.size();
  vector<double> x(rowsA, 1.0);
  vector<double> y(rowsA, 0.0);
  vector<double> y_verified(rowsA, 0.0);
  StandardSpMV(ColIndexA, RowPtrA, ValueA, x, y_verified);
  for(int i = 0; i < TESTS_FOR_SINGLE_FILE; i++)
    {
      fill(y.begin(), y.end(), 0.0);
      double MergeTime = 0.0;
      double DiagonalTime = 0.0;
      TimeVar totalStart = timeNow();
      MergePath(RowPtrA, ColIndexA, ValueA, x, y, MergeTime, DiagonalTime);
      TimeVar totalEnd = timeNow();
      double totalTime = duration(totalEnd-totalStart);
      CompareVectors(y, y_verified);
      //Filename:nnzA:DiagonalTime:MergeTime:TotalTime
      outputFile << filename << "\t" << nnzA << "\t" << DiagonalTime << "\t" << MergeTime << "\t" << totalTime << endl;
    }
  outputFile.close();
}
inline void GetMatrix(char* filename, vector<int> &csrRowPtrA, vector<int> &csrColIndexA, vector<VALUE_TYPE> &csrValueA)
{  
  int columnsA, nnzA_mtx, nnzA, rowsA;
  //Based on example by MM
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  f = fopen(filename, "r");
  int isInteger = 0, isReal = 0, isSymmetric = 0;
  if (f == NULL)
    exit(1);

  if ( mm_read_banner(f, &matcode) != 0)
    {
      cout << "Could not process Matrix Market banner." << endl;
      exit(1);
    }
  if ( mm_is_complex( matcode ) )
    {
      cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
      exit(1);
    }
  if ( mm_is_real ( matcode) )     { isReal = 1; }
  if ( mm_is_integer ( matcode ) ) { isInteger = 1; }
  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &rowsA, &columnsA, &nnzA_mtx)) !=0)
    {
      exit(1);
    }
  if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
      isSymmetric = 1;
    }

  //Stores how many nnzs are in a given row ([row])
  vector<int> temp_csrRowPtrA(rowsA+1, 0);
  //Hold all row indexes, colun indexes, and values corresponding to the [l] they were entered in 
  //i.e. temp_csrValueA[l] is the value at row = temp_csrRowIndexA[l] and col = temp_csrRowIndexA[l]
  //l represents a number between 0-l (which is the # of nnz)
  vector<int> temp_csrRowIndexA(nnzA_mtx); 
  vector<int> temp_csrColIndexA(nnzA_mtx);
  vector<VALUE_TYPE> temp_csrValueA(nnzA_mtx);

  #pragma omp parallel for default(none) firstprivate(f, nnzA_mtx, isInteger, isReal) shared(temp_csrRowPtrA, temp_csrRowIndexA, temp_csrColIndexA, temp_csrValueA)
  for (int k = 0; k < nnzA_mtx; k++)
    {
      int i, j;
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
      else
	{
	  temp_csrValueA.at(k) = int_value;
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
}
inline void StandardSpMV(vector<int> csrColIndexA, vector<int> csrRowPtrA, vector<VALUE_TYPE> csrValueA, vector<double> x, vector<double> &y_verified)
{
  int sizeA = csrRowPtrA.size();
  double totalData;
  for(int i = 0; i < sizeA-1; i++, totalData = 0.0)
    {
      for(int j = csrRowPtrA.at(i); j <  csrRowPtrA.at(i+1); j++)
	{
	  totalData += csrValueA.at(j) * x.at(csrColIndexA.at(i));
	}
      y_verified.at(i) = totalData;
    }

}
inline void CompareVectors(vector<double> y1, vector<double> y2)
{
  if(y1 == y2)
    {
      cout << "Success!" << endl;
    }
  else
    {
      cout << "Failure." << endl;
    }
}
