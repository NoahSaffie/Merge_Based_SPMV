/*
  Supply the program with a single .mtx file through command line
  or you can supply it a .txt file that has the following:
  path to directory with .mtx fiiles as first line
  every subsequent line is a filename for an mtx.
  If file names in .txt already include .mtx extension then change REQUIRES_MTX_EXTENSION below to 0
 */
#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <malloc.h>
extern "C"{
#include "mmio.h"
}

using namespace std;

//Credit to: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
//Also matches this template/default https://www.codeproject.com/Articles/4795/C-Standard-Allocator-An-Introduction-and-Implement
//http://man7.org/linux/man-pages/man3/posix_memalign.3.html
//https://msdn.microsoft.com/en-us/library/8z34s9c6.aspx
template <typename T, std::size_t N = 16>
class AlignmentAllocator {
public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T * pointer;
  typedef const T * const_pointer;
  typedef T & reference;
  typedef const T & const_reference;

  public:
  inline AlignmentAllocator () throw () { }

  template <typename T2>
  inline AlignmentAllocator (const AlignmentAllocator<T2, N> &) throw () { }
  inline ~AlignmentAllocator () throw () { }
  inline pointer address (reference r) { return &r; }
  inline const_pointer address (const_reference r) const { return &r; }
  inline pointer allocate (size_type n) { return (pointer)aligned_alloc(N, n*sizeof(value_type)); }
  inline void deallocate (pointer p, size_type) { free(p); }
  inline void construct (pointer p, const value_type & wert) { new (p) value_type (wert); }
  inline void destroy (pointer p) { p->~value_type (); }
  inline size_type max_size () const throw () { return size_type (-1) / sizeof (value_type); }
  template <typename T2>
  struct rebind {
    typedef AlignmentAllocator<T2, N> other; 
  };

  bool operator!=(const AlignmentAllocator<T,N>& other) const  {
    return !(*this == other);
  }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const AlignmentAllocator<T,N>& other) const {
    return true;
  }
};
#define showFailure 1
#define ALIGNMENT 32
#define REQUIRES_MTX_EXTENSION 1

//Allows easy switch of using alignment or now
#define ALIGN_DOUBLE
//#define ALIGN_DOUBLE , AlignmentAllocator<double, ALIGNMENT>
//Credit to: https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()
struct Coord{int x;int y;};


inline void DetermineCoordinate(int diagonal, vector<int>csrRowPtrA, int sizeB, Coord &point);
inline void MergePath(vector<int> csrRowPtrA, vector<int> csrColIndex, vector<double ALIGN_DOUBLE> csrValueA, vector<double ALIGN_DOUBLE> x, vector<double> &y, double* totalTime);
inline void StandardSpMV(vector<int> csrRowPtrA, vector<int> csrColIndexA, vector<double ALIGN_DOUBLE> csrValueA, vector<double ALIGN_DOUBLE> x, vector<double> &y_verified);
inline int  GetMatrix(string filename, vector<int> &csrRowPtrA, vector<int> &csrColIndexA, vector<double ALIGN_DOUBLE> &csrValueA);
inline void CompareVectors(vector<double> y1, vector<double> y2);
inline void RunTests(string filename);
int main(int argc, char* argv[])
{
  /*
    Looks for a Command Line Arguement that is either a single .mtx file
    or a .txt that contains a list of .mtx files, with one per line and nothing else
   */
  
  ofstream outputFile;
  outputFile.open("output.csv", ios::out);
  outputFile << "Matrix,Non-Zeros,Average time to find Coordinate along diagonal,Average Path Traversal and Matrix Multipication,Average total time for MergePath Function,Time to get Matrix,Tests Ran, Average time for set instructions, Average Time for mul instruction, Average time for Summing instructions, Count (SIMD Loops)" << endl; //Header
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
inline void MergePath(vector<int> csrRowPtrA, vector<int> csrColIndex, vector<double ALIGN_DOUBLE> csrValueA, vector<double ALIGN_DOUBLE> x, vector<double> &y, double* totalTime)
{
  /*
    Changes: 
    Abandoned ListB entirely, can accomplish its use by simply having a 'y' value when finding coordinates 
    Simplified the section about a row spanning across more than one team, no needed overhead like before
    Calculate IPT slightly different (In a more base logical way [Possible I just miss why the other way is better])
    Simpler basis for determing diagonals - Works entirely based the Items per thread
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
 
  int count = 0;
  double set = 0.0;
  double sum = 0.0;
  double mul = 0.0;
  int num_of_threads = omp_get_num_threads(); 
  #pragma omp parallel for num_threads(num_of_threads) schedule(static) default(none) firstprivate(num_of_threads, x, csrValueA, csrColIndex, csrRowPtrA, set, sum, mul, count) shared(y, totalTime)
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
      TimeVar mergeStart = timeNow();
      int loopY;
      double totalData = 0.0;
      double results[4];
      __m256d allZeros = _mm256_setzero_pd();
      for(; startCoord.x <= endCoord.x; ++startCoord.x, totalData = 0.0)
	{   
	  int maxYforLoop = min(csrRowPtrA.at(startCoord.x+1)-1, endCoord.y);
	  //#pragma omp simd
	  for(loopY = startCoord.y; loopY <= maxYforLoop; ++loopY)
	    {
	      //Diagonal[0], Merge[1], func[2], Matrix[3], SIMD_SET[4], SIMD_MUL[5], SIMD_ADD[6], SIMD_PERM[7]
	      if(maxYforLoop-loopY >= 8)
		{
		  count++;
		  //Get values (Will need to be cast to double)
		  //Use setr (reverse) so that first value supplied is the first value in the mem/register
		  TimeVar s1 = timeNow();
		  //__m256d values1 = _mm256_setr_pd(csrValueA.at(loopY), csrValueA.at(loopY+1),csrValueA.at(loopY+2),csrValueA.at(loopY+3));
		  //__m256d values2 = _mm256_setr_pd(csrValueA.at(loopY+4),csrValueA.at(loopY+5),csrValueA.at(loopY+6),csrValueA.at(loopY+7));
		  __m256d values1 = _mm256_loadu_pd(&(csrValueA.at(loopY)));
		  __m256d values2 = _mm256_loadu_pd(&(csrValueA.at(loopY+4)));
		  //Get the x-values
		  __m256d x1 = _mm256_setr_pd(x.at(csrColIndex.at(loopY)),x.at(csrColIndex.at(loopY+1)),x.at(csrColIndex.at(loopY+2)),x.at(csrColIndex.at(loopY+3)));
		  __m256d x2 = _mm256_setr_pd(x.at(csrColIndex.at(loopY+4)),x.at(csrColIndex.at(loopY+5)),x.at(csrColIndex.at(loopY+6)),x.at(csrColIndex.at(loopY+7)));
		  TimeVar s2 = timeNow(); set+=duration(s2-s1);
		  //Multiply
		  s1 = timeNow();
		  values1 = _mm256_mul_pd(values1, x1);
		  values2 = _mm256_mul_pd(values2, x2);
		  s2 = timeNow(); mul+=duration(s2-s1);
		  //Sum all results

		  s1 = timeNow();
		  __m256d valuesSum = _mm256_hadd_pd(values1, values2);
		  values1 = _mm256_permute2f128_pd(valuesSum, allZeros, 19); //00010011b Get upper 128 of first param, and upper 128 of second param
		  values2 = _mm256_permute2f128_pd(valuesSum, allZeros, 2); //00000010b Get lower 128 of first param, and lower 128 of second param
		  valuesSum = _mm256_hadd_pd(values1, values2);
		  
		  //Final add
		  values1 = _mm256_permute2f128_pd(valuesSum, allZeros, 19);
		  valuesSum = _mm256_hadd_pd(values1, allZeros);
		  s2 = timeNow(); sum+=duration(s2-s1);
		  //Access the final result
		  _mm256_storeu_pd(results, valuesSum);
		  totalData+=(results[2]+results[3]);
		  loopY+=7; //Other ++ is still in loop header
		}
	      else
		{
		  totalData += (csrValueA.at(loopY) * x.at(csrColIndex.at(loopY)));
		}
	    }
	  startCoord.y = loopY;
          #pragma omp critical(OutputUpdate)
	  y.at(startCoord.x) = y.at(startCoord.x) +  totalData;
	}
      TimeVar mergeEnd = timeNow();
#pragma omp critical(TimeOutput)
      {
	totalTime[0]+= duration(diagonalEnd - diagonalStart);
	totalTime[1]+= duration(mergeEnd-mergeStart);
	totalTime[5]+= mul;
	totalTime[4]+= set;
	totalTime[6]+= sum;
	totalTime[7] = count*1.0;
      }
    }
#pragma omp barrier
}
inline void RunTests(string filename)
{
  /* Check if we are going to use int or double for values */
  MM_typecode matcode;
  FILE *f;
  int columns, rowsA, nnzA, TESTS_FOR_SINGLE_FILE, ret_code;
  f = fopen(filename.c_str(), "r");
  if (f == NULL)
    return;
  if (mm_read_banner(f, &matcode) != 0){cout << "Could not process Matrix Market banner." << endl; return;}
  if (mm_is_complex(matcode)){ cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl; return;}
  if ((ret_code = mm_read_mtx_crd_size(f, &rowsA, &columns, &nnzA)) !=0){ return;}
  fclose(f);

  //File I/O
  ofstream outputFile;
  outputFile.open("output.csv", ios::out | ios::app);
  string separator(","); //comma for CSV
  //outputFile << "Matrix" << separator << "Non-zeros" << separator << "Time for Diagonal fucntion" << separator << "Time for Merge path Loop" << separator << "Time for total Merge Function" << endl;
  
  //Vector Declarations/Inits
  vector<int> RowPtrA;
  vector<int> ColIndexA;
  double timeTotal[8] = {0.0}; //Diagonal[0], Merge[1], func[2], Matrix[3], SIMD_SET[4], SIMD_MUL[5], SIMD_ADD[6], SIMD_PERM[7] 
  vector<double ALIGN_DOUBLE> x(columns, 1.0);
  vector<double> y(rowsA, 0.0);
  vector<double> y_verified(rowsA, 0.0);
  TESTS_FOR_SINGLE_FILE = max(5, 16000000/nnzA);
  vector<double ALIGN_DOUBLE> ValueA;
  //Start processing matrix - Base results and CSR only need to be found once.
  TimeVar matrixStart = timeNow();
  nnzA = GetMatrix(filename, RowPtrA, ColIndexA, ValueA);
  TimeVar matrixEnd = timeNow();
  timeTotal[3]+= duration(matrixEnd-matrixStart);
  if(columns < 0) //Returned a negative which is an error
    {
      outputFile.close();
      return;
    }
  StandardSpMV(RowPtrA, ColIndexA, ValueA, x, y_verified);
  //Run the Mergebased Spmv function multiple times to find a more accurate average time value
  for(int i = 0; i < TESTS_FOR_SINGLE_FILE; i++)
    {
      fill(y.begin(), y.end(), 0.0);     
      TimeVar singleFuncStart = timeNow();
      MergePath(RowPtrA, ColIndexA, ValueA, x, y, timeTotal);
      TimeVar singleFuncEnd = timeNow();
     
      CompareVectors(y, y_verified);
      timeTotal[2]+=duration(singleFuncEnd-singleFuncStart);
    }
  //Change filename to only include file not path
  outputFile << filename.substr(filename.find_last_of("/\\")+1) << separator << nnzA << separator << timeTotal[0]/TESTS_FOR_SINGLE_FILE << separator << timeTotal[1]/TESTS_FOR_SINGLE_FILE << separator << timeTotal[2]/TESTS_FOR_SINGLE_FILE << separator << timeTotal[3]<< separator << TESTS_FOR_SINGLE_FILE << separator << timeTotal[4]/timeTotal[7] << separator << timeTotal[5]/timeTotal[7] << separator << timeTotal[6]/timeTotal[7] << separator << timeTotal[7] <<endl;
  outputFile.close();
}
inline int GetMatrix(string filename, vector<int> &csrRowPtrA, vector<int> &csrColIndexA, vector<double ALIGN_DOUBLE> &csrValueA)
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

  if (mm_read_banner(f, &matcode) != 0){ cout << "Could not process Matrix Market banner." << endl; return -2; }
  //cout << "Matcode: " << matcode << endl;
  //cout << mm_typecode_to_str(matcode) << endl;
  if(mm_is_pattern(matcode)){ isBinary = 1; }
  if(mm_is_complex(matcode)){ cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl; return -3; }
  if(mm_is_real(matcode)){ isReal = 1; }
  if(mm_is_integer(matcode )){ isInteger = 1; }
  /* find out size of sparse matrix .... */
  if((ret_code = mm_read_mtx_crd_size(f, &rowsA, &columnsA, &nnzA_mtx)) !=0){ return -4; }
  if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)){ isSymmetric = 1; }
  vector<int> temp_csrRowPtrA(rowsA+1, 0);
  vector<int> temp_csrRowIndexA(nnzA_mtx); 
  vector<int> temp_csrColIndexA(nnzA_mtx);
  vector<double> temp_csrValueA(nnzA_mtx);

  #pragma omp parallel for default(none) firstprivate(f, nnzA_mtx, isReal, isInteger, isBinary) shared(temp_csrRowPtrA, temp_csrRowIndexA, temp_csrColIndexA, temp_csrValueA)
  for (int k = 0; k < nnzA_mtx; k++)
    {
      int i =-1, j = -1;
      double double_value;
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
	  temp_csrValueA.at(k) = (double)int_value;
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
  csrValueA = vector<double ALIGN_DOUBLE>(nnzA);
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
  return nnzA;
}
inline void StandardSpMV(vector<int> csrRowPtrA, vector<int> csrColIndexA, vector<double ALIGN_DOUBLE> csrValueA, vector<double ALIGN_DOUBLE> x, vector<double> &y_verified)
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
      if(showFailure)
	{
	  for(unsigned i = 0; i < y1.size(); i++)
	    {
	      //Comparing doubles directly will give incorrect answer due to rounding
	      if(y1.at(i)/y2.at(i)>= 1.001 || y1.at(i)/y2.at(i) <= .999)
		{
		  cout << "Failure." <<  endl;
		  cout << "At line: " << i << " of size " << y1.size() << endl;
		  cout << "\ty1: " << y1.at(i) << "\ty2: " << y2.at(i) << endl;
		}
	    }
	}
    }
}
