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
  if(argc < 2)
    {
      filename = argv[1];
      cout << "--------------" << filename << "--------------" << endl;
    }
  //Read matrix from mtx file
  //Matrix Market == MM
  //Based on example by MM
  int ret_code;
  MM_typecode matcode;
  FILE *f;

  int isInteger = 0, isReal = 0, isSymmetric = 0;

  // load matrix
  if ((f = fopen("1138_bus.mtx", "r")) == NULL)
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

  if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
  if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &rowsA, &columnsA, &nnzA_mtx)) !=0)
    exit(1);

  //Check if symmetric ---Might not need for my purposes but will keep for now
  if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
      isSymmetric = 1;
      //cout << "symmetric = true" << endl;
    }

  //Stores how many nnzs are in a given row ([row])
  int *temp_csrRowPtrA = (int *)malloc((rowsA+1) * sizeof(int));
  memset(temp_csrRowPtrA, 0, (rowsA+1) * sizeof(int)); //Make empty/all zero's to start

  //Hold all rox indexes, colun indexes, and values corresponding to the [l] they were entered in 
  //i.e. csrvalA_tmp[l] is the value at row = csrRowIndexA_tmp[l] and col = csrRowIndexA_tmp[l]
  //l represents a number between 0-l (which is the # of nnz)
  int *temp_csrRowIndexA = (int *)malloc(nnzA_mtx * sizeof(int));
  int *temp_csrColIndexA = (int *)malloc(nnzA_mtx * sizeof(int));
  VALUE_TYPE *temp_csrValueA = (VALUE_TYPE *)malloc(nnzA_mtx * sizeof(VALUE_TYPE));

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  //Assumptions:
  //tmp == temp
  //idx == index

  //Can likely do OpenMP if the scanf part is set as critical
  #pragma omp parallel for default(none) firstprivate(f, nnzA_mtx, isInteger, isReal) shared(temp_csrRowPtrA, temp_csrRowIndexA, temp_csrColIndexA, temp_csrValueA)
  for (int k = 0; k < nnzA_mtx; k++)
    {
      int i, j;
      VALUE_TYPE double_value;
      int int_value;
      //Can only scan in safely one at a time, otherwise might break future scans
      #pragma omp critical
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
  //If we had a file, close it?
  if (f != stdin)
    fclose(f);

  //Below taken from CSR5 code with slight alterations
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

  //-----Check OpenMP will work with Symmetry piece-----
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

  //Need to modify csrRowPtrA to not include the intial 0 (since it will throw off rest of program)
  
  //init vectors
  vector<double> x(rowsA, 1.0);
  vector<double> y(rowsA, 0.0);
  
  
  //MergePathSearch
  MergePath(rowsA, nnzA, &(csrRowPtrA[1]), csrColIndexA, csrValueA, x, y);
  //Final Clean-up
  free(csrColIndexA);
  free(csrValueA);
  free(csrRowPtrA);
  return 0;
}
inline void DetermineCoordinate(int diagonal, int* csrRowPtrA, int sizeA, int startX, Coord &point)
{
  //int num_threads = omp_get_num_threads(); //May need for some optimization
  //maximum iterations is same as Items Per Thread (In reality should always be less)
  int x = startX;
  for(; x<sizeA && ((diagonal-x) > csrRowPtrA[x]); ++x);

  //Output
  if(x >= sizeA) //Try to resolve in a different way down road when expand above in optimization attempt
    {
      x = sizeA-1;
    }
  point.x = x;
  point.y = diagonal-x;  
}
inline void MergePath(int sizeA, int sizeB, int* csrRowPtrA, int* csrColIndex, VALUE_TYPE* csrValueA, vector<double> x, vector<double> &y)
{  
  int num_of_threads = omp_get_num_threads();
  //There are 2 diagonals for every thread/team
  //Only two are used exactly once (first and last one)
  //Others are a shared diagonal
  //So we really only need 1 diagonal per thread plus 1 extra
  Coord* coordList = (Coord*)malloc(sizeof(Coord)*(num_of_threads+1));
  int totalPathLength = sizeA+sizeB;
  int IPT = totalPathLength/num_of_threads; //Items Per Thread
  //Create coordList
  //Always has a 0,0 start
  coordList[0] = {0, 0};
  //Cannot use OpenMP here (Wouldn't want to much aanyways), but will try to utilize within funciton itself
  for(int j = 0; j < num_of_threads; ++j)
    {
      DetermineCoordinate(IPT*(j+1), csrRowPtrA, sizeA, coordList[j].x, coordList[j+1]);
    }
  //OPEN MP
  //-----------Requires editing about how y is shared/accessed---------------
  //Don't/Can't use collapse on outer loop ----------------investigate for inner ones -----------------
#pragma omp parallel for num_threads(num_of_threads) default(none) firstprivate(num_of_threads, coordList, x, csrValueA, csrColIndex, csrRowPtrA) shared(y, cout)
  for(int i = 0; i < num_of_threads; i++)
    {
      Coord startCoord = coordList[i];
      Coord endCoord = coordList[i+1];
      double totalData;
      //Collect info from areas
      //This needs to change should just look like what you would do to draw the path, but it does math along the way
      //One iteration is completing one row (or all it can for a team)
      for(; startCoord.x <= endCoord.x; ++startCoord.x)
	{
	  //startCoord.y  is the value(an index) of listB
	  //Sum the row
	  totalData = 0.0;
	  //cout << "Start x: " << startCoord.x << endl;
	  //-----------------Should be able to utilize reduction here -------------------
	  for(; startCoord.y <= endCoord.y && csrRowPtrA[startCoord.x] > startCoord.y; ++startCoord.y)
	    {
	      //x[csrColIndex[startCoord.y]] is a part of our selective dot product (Ignoring 0 sums)
	      //We are doing row*column, we get row value from:  Value<row, column> where row is set by outer loop, and column is startCood.y (this is as if in Matrix)
	      //For the column we want the value in X that corresponds to the column  in the Matrix we are working in so x[csrColIndex[startCoord.y]] does this
	      totalData += (csrValueA[startCoord.y] * x.at(csrColIndex[startCoord.y])); //Double check if x or y. This is just a starting place 
	    }
	  //Is startCoord.x because we summed the total for a row, and startcoord.x represents a row
	  //Now we want to set a vlaue in the output Vector/Matrix corresponding to row level/number/index
	  //Believe that doing += (as long as y starts as all zeros), should take care of the issue of a row going across teams
	  //When one team finishes its sum for the part of the row it can see it adds it to that index, and when the other finishs its part it will do same
	  //----------------------------May require Critical or similar (Since if two teams tried to add to it at same time the latter would be the only result) --------------
	  y.at(startCoord.x) = y.at(startCoord.x) +  totalData;
	}	  
    }
  free(coordList);
}
