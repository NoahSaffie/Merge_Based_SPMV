#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
extern "C"
{
#include "mmio.h"
}
using namespace std;
#define VALUE_TYPE double


struct Coord
{
  int x;
  int y;
};

inline void DetermineCoordinate(int diagonal, int* csrRowPtrA, int sizeA, int startX, Coord &point);
inline void MergePath(int sizeA, int sizeB, int* csrRowPtrA, int* csrColIndex, VALUE_TYPE* csrValueA, vector<double> x, vector<double> &y);
inline void StandardSpMV(int* csrColIndexA, int* csrRowPtrA, VALUE_TYPE* csrValueA, vector<double> x, vector<double> &y_verified, int sizeA);
inline void CompareVectors(vector<double> y1, vector<double> y2);

int main(int argc, char* argv[])
{
  int rowsA, columnsA, nnzA, nnzA_mtx;
  //A is a matrix, nnz == non-zeros

  //Basic CSR
  int *csrRowPtrA;
  //RowPtr Array
  int *csrColIndexA;
  //Column Index Array
  VALUE_TYPE *csrValueA;
  //Values Array
  char  *filename;
  if(argc == 2)
    {
      filename = argv[1];
      cout << "--------------" << filename << "--------------" << endl;
    }
  else
    {
      cout << "Failed to recongize or recieve a single filename (.mtx) as a command line argument)" << endl;
    }
  //Read matrix from mtx file
  //Matrix Market == MM
  //Based on example by MM
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  f = fopen(filename, "r");
  int isInteger = 0, isReal = 0, isSymmetric = 0;

  // load matrix
  if (f == NULL)
    return -1;

  if ( mm_read_banner(f, &matcode) != 0)
    {
      cout << "Could not process Matrix Market banner." << endl;
      return -2;
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
      exit(1);
    }
  if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
      isSymmetric = 1;
    }

  //Stores how many nnzs are in a given row ([row])
  int *temp_csrRowPtrA = (int *)malloc((rowsA+1) * sizeof(int));
  memset(temp_csrRowPtrA, 0, (rowsA+1) * sizeof(int)); //Make empty/all zero's to start since will be using incrementing

  //Hold all row indexes, colun indexes, and values corresponding to the [l] they were entered in 
  //i.e. temp_csrValueA[l] is the value at row = temp_csrRowIndexA[l] and col = temp_csrRowIndexA[l]
  //l represents a number between 0-l (which is the # of nnz)
  int *temp_csrRowIndexA = (int *)malloc(nnzA_mtx * sizeof(int));
  int *temp_csrColIndexA = (int *)malloc(nnzA_mtx * sizeof(int));
  VALUE_TYPE *temp_csrValueA = (VALUE_TYPE *)malloc(nnzA_mtx * sizeof(VALUE_TYPE));

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
      temp_csrRowPtrA[i]++;
      temp_csrRowIndexA[k] = i;
      temp_csrColIndexA[k] = j;
      if(isReal)
	{
	  temp_csrValueA[k] = double_value;
	}
      else
	{
	  temp_csrValueA[k] = int_value;
	}
    }
  fclose(f);

  //Below taken from CSR5 code with many alterations
  if (isSymmetric)
    {
      #pragma omp parallel for default(none) firstprivate(nnzA_mtx) shared(temp_csrRowIndexA, temp_csrRowPtrA, temp_csrColIndexA)
      for (int i = 0; i < nnzA_mtx; i++)
	{
	  if (temp_csrRowIndexA[i] != temp_csrColIndexA[i])
	    temp_csrRowPtrA[temp_csrColIndexA[i]]++;
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
      new_val = temp_csrRowPtrA[i];
      temp_csrRowPtrA[i] = old_val + temp_csrRowPtrA[i-1];
      old_val = new_val;
    }
  //Will enlarge if we are doing a symmetric matrix
  nnzA = temp_csrRowPtrA[rowsA];
  
  csrRowPtrA = (int *)malloc((rowsA+1) * sizeof(int)); //Init the csrRowPtr for Matrix A (this was done earlier automatically for out tmp versions in function call)
  memcpy(csrRowPtrA, temp_csrRowPtrA, (rowsA+1) * sizeof(int)); //copy the value from temp to this array
  memset(temp_csrRowPtrA, 0, (rowsA+1) * sizeof(int)); //clear the old one

  csrColIndexA = (int *)malloc(nnzA * sizeof(int)); //init
  csrValueA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE)); //init
  //this will order the (potentially) unordered data that is currently in the temp arrays
  //however it doesn't completely order it, columnIndexes/Value Indexes will not be in exact order
  //But it shouldn't matter everything is in correct row sections and the column index, and value index are still a valid pair

#pragma omp parallel for default(none) shared(csrColIndexA, csrValueA, temp_csrRowPtrA) firstprivate(temp_csrRowIndexA, temp_csrColIndexA, temp_csrValueA, isSymmetric, csrRowPtrA, nnzA_mtx)
  for (int i = 0; i < nnzA_mtx; i++)
    {
      int tempRowIndexAti = temp_csrRowIndexA[i];
      int offset = csrRowPtrA[tempRowIndexAti] + temp_csrRowPtrA[tempRowIndexAti]++;

      csrColIndexA[offset] = temp_csrColIndexA[i];
      csrValueA[offset] = temp_csrValueA[i];
      //Not on the diagonal (like y = -x) that the symmetry is defined around
      if (isSymmetric && tempRowIndexAti != temp_csrColIndexA[i])
	{
	  //Symmetry piece
	  offset = csrRowPtrA[temp_csrColIndexA[i]] + temp_csrRowPtrA[temp_csrColIndexA[i]]++;
	  csrColIndexA[offset] = temp_csrRowIndexA[i];
	  csrValueA[offset] = temp_csrValueA[i];
	}
    }
  // free temp space
  free(temp_csrColIndexA);
  free(temp_csrValueA);
  free(temp_csrRowIndexA);
  free(temp_csrRowPtrA);

  //Use the now init CSR
  //csrColIndexA
  //csrValueA
  //csrRowIndexA
  //csrRowPtrA

  //init vectors
  vector<double> x(rowsA, 1.0);
  vector<double> y(rowsA, 0.0);

  vector<double> y_verified(rowsA, 0.0);
  StandardSpMV(csrColIndexA, csrRowPtrA, csrValueA, x, y_verified, rowsA+1);
  //MergePathSearch
  //Need to modify csrRowPtrA to not include the intial 0 (since it will throw off rest of program)
  //Tell Merge path to ignore first element by passing it address of element[1] as start of array
  //Size is rowsA since previously our RowPtrA was rowsA+1 to fit a extra zero at beginning or the end (depending on if temp or final version)
  MergePath(rowsA, nnzA, &(csrRowPtrA[1]), csrColIndexA, csrValueA, x, y);

  CompareVectors(y, y_verified);
  //Final Clean-up 
  free(csrColIndexA);
  free(csrValueA);
  free(csrRowPtrA);
  return 0;
}
inline void DetermineCoordinate(int diagonal, int* csrRowPtrA, int sizeA, int startX, Coord &point)
{
  /*
    Changes:
    Use the already known starting coordinate (the x-value specifically) for this section to find the end coordinate much quicker
    Use a loop to work directly up to end coordinate instead of using a pivot to bring x_min and x_max closer, until together determine endCoord
   */
  //int num_threads = omp_get_num_threads(); //May need for some optimization
  //maximum iterations is same as Items Per Thread (In reality should always be less)
  int x = startX;
  int y = diagonal-x;
  //schedule will be monotonic - increasing iteration order
  //might need thread cancellization - Want to be able to "cancel" a single thread and disregard all iterations it was assigned
  //And when using the cancellization dynamic is gonna be our best choice
  for(; x<sizeA && (y > csrRowPtrA[x]); ++x, y = diagonal-x);
  //May want to play around with the conditioning to avoid false positives from concurrency
  //Such as the break condition being finding the x,y where the y is less than RowPtr[x] but @ the coordinate right before it the y > RowPtr[x]
  //Use shortcircuting with this to avoid unnecessary overhead when possible

  //Another nice option would be to keep track of a x_max that is determined by and updated by threads that have hit a invalid coordinate we know to be past the correct point
  //And then reassigning the threads to the now know area
  //---------------Hoping to avoid this by using monotonic scheduling and hopefully something else to have the concurrent iterations be in very close increasing iteration order----------
  //so focus on solving that
  
  //Output
  if(x >= sizeA) 
    {
      x = sizeA-1;
    }
  point.x = x;
  point.y = y;  
}
inline void MergePath(int sizeA, int sizeB, int* csrRowPtrA, int* csrColIndex, VALUE_TYPE* csrValueA, vector<double> x, vector<double> &y)
{
  /*
    Changes: 
    Abandoned ListB entirely, can accomplish its use by simply having a 'y' value when finding coordinates 
    Simplified the section about a row spanning across more than one team, no needed overhead like before
    Calculate IPT slightly different (In a more base logical way [Possible I just miss why the other way is better])
    Determine all coordinates before traversing the path at a team/thread level, this allows all the start/end coordinates of teams to be found much quicker (stored more condensly too!)
    Simpler basis for determing diagonals - Works entirely based the Items per thread
   */
  /*
    Drawbacks/Considerations:
    Workload can become unbalanced as Thread # approaches Path Length (which should be a rare issue since using with large data sets)
        Still should examine if we can improve this niche (either by checking if PathLength is close to # of threads and respond with a different IPT ditermination, or a new basis for IPT for all cases)
    Checking if OpenMP is actually enabled and not operating under the assumption it is
   */
  /*
    Notes:
    startCoord.y  is the value(an index) of listB
    IPT - Items Per Thread
    CoordList - There are 2 diagonals for every thread/team, only two are used exactly once (first and last one), others are a shared diagonal, so only need 1 diagonal per thread plus 1 extra
        Must always have a starting coordinate of (0,0) as that would be the first starting diagonal and the diagonal only crosses a single point (0,0)
    Getting totalData - x[csrColIndex[startCoord.y]] is a part of our selective dot product (Ignoring 0 sums)
        We are doing row*column, we get row value from:  Value<row, column> where row is set by outer loop, and column is startCood.y (this is as if in Matrix)
        For the column we want the value in X that corresponds to the column  in the Matrix we are working in so x[csrColIndex[startCoord.y]] does this
   */
  int num_of_threads = omp_get_num_threads();
  Coord* coordList = (Coord*)malloc(sizeof(Coord)*(num_of_threads+1));
  int totalPathLength = sizeA+sizeB;
  int IPT = totalPathLength/num_of_threads; //Change by +1 if not easily divisible ?
  coordList[0] = {0, 0};
  //Cannot use OpenMP here, but can utilize within funciton itself
  for(int j = 0; j < num_of_threads; ++j)
    {
      DetermineCoordinate(IPT*(j+1), csrRowPtrA, sizeA, coordList[j].x, coordList[j+1]);
    }
  #pragma omp parallel for num_threads(num_of_threads) default(none) firstprivate(num_of_threads, coordList, x, csrValueA, csrColIndex, csrRowPtrA) shared(y)
  for(int i = 0; i < num_of_threads; i++)
    {
      Coord startCoord = coordList[i];
      Coord endCoord = coordList[i+1];
      double totalData;
      //Collect info from areas
      for(; startCoord.x <= endCoord.x; ++startCoord.x, totalData = 0.0)
	{
	  //Sum the row
	  for(; startCoord.y <= endCoord.y && csrRowPtrA[startCoord.x] > startCoord.y; ++startCoord.y)
	    {
	      totalData += (csrValueA[startCoord.y] * x.at(csrColIndex[startCoord.y]));
	    }
          #pragma omp critical(OutputUpdate)
	  y.at(startCoord.x) = y.at(startCoord.x) +  totalData;
	}	  
    }
  free(coordList);
}
inline void StandardSpMV(int* csrColIndexA, int* csrRowPtrA, VALUE_TYPE* csrValueA, vector<double> x, vector<double> &y_verified, int sizeA)
{
  double totalData;
  for(int i = 0; i < sizeA-1; i++, totalData = 0.0)
    {
      for(int j = csrRowPtrA[i]; j <  csrRowPtrA[i+1]; j++)
	{
	  totalData += csrValueA[j] * x.at(csrColIndexA[i]);
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
  /*
  for(int i = 0; i < y1.size(); i++)
    {
      cout << "y1: " << y1.at(i) << "\ty2: " << y2.at(i) << endl;
    }
  */
}
