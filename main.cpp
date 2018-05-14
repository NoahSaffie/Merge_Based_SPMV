#include <iostream>
#include <cstdlib>
#include "mmio.h"


#define VALUE_TYPE double
int main()
{
  int rowsA, columnsA, nnzA;
  //A is a matrix, nnz == non-zeros

  //Basic CSR
  int *csrRowPtrA;
  //RowPtr Array
  int *csrColIndexA;
  //Column Index Array
  VALUE_TYPE *csrValueA;
  //Values Array
  int argi = 1;

  char  *filename;
  if(argc > argi)
    {
      filename = argv[argi];
      argi++;
    }
  cout << "--------------" << filename << "--------------" << endl;

  //Read matrix from mtx file
  //Matrix Market == MM
  //Based on example by MM
  int ret_code;
  MM_typecode matcode;
  FILE *f;

  int nnzA_mtx; //Returned Amount of non-zeros
  int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

  // load matrix
  if ((f = fopen(filename, "r")) == NULL)
    return -1;

  if (mm_read_banner(f, &matcode) != 0)
    {
      cout << "Could not process Matrix Market banner." << endl;
      return -2;
    }

  if ( mm_is_complex( matcode ) )
    {
      cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
      return -3;
    }

  if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
  if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
  if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
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
  for (int i = 0; i < nnzA_mtx; i++)
    {
      int i, j;
      double double_value;
      int int_value;

      //If doubles
      if (isReal)
	fscanf(f, "%d %d %lg\n", &i, &j, &double_value);
      //If Ints
      else if (isInteger)
        {
	  fscanf(f, "%d %d %d\n", &i, &j, &int_value);
	  fval = ival;
        }

      // adjust from 1-based to 0-based
      i--;
      j--;

      //Add the value/indexes to respective arrays
      temp_csrRowPtrA[i]++;
      temp_csrRowIndexA[i] = i;
      temp_csrColIndexA[i] = j;
      if(isReal)
	{
	  temp_csrValueA[i] = double_value;
	}
      else
	{
	  temp_csrValueA[i] = int_value;
	}
    }
  //If we had a file, close it?
  if (f != stdin)
    fclose(f);

  //Below taken from CSR5 code with slight alterations
  if (isSymmetric)
    {
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
  old_val = temp_csrRowPtrA[0];
  temp_csrRowPtrA[0] = 0;
  for (int i = 1; i <= rowsA; i++)
    {
      new_val = temp_csrRowPtrA[i];
      temp_csrRowPtrA[i] = old_val + temp_csrRowPtrA[i-1];
      old_val = new_val;
    }

  nnzA = temp_csrRowPtrA[rowsA]; //num of non-zeros in Matrix A
  //_mm_malloc is a special version of malloc by intel that aligns memory, the second parameter is the alignment size, and in this case it seems the program is using a special one from its other library
  csrRowPtrA = (int *)malloc((rowsA+1) * sizeof(int)); //Init the csrRowPtr for Matrix A (this was done earlier automatically for out tmp versions in function call)
  memcpy(csrRowPtrA, temp_csrRowPtrA, (rowsA+1) * sizeof(int)); //copy the value from temp to this array
  memset(temp_csrRowPtrA, 0, (rowsA+1) * sizeof(int)); //clear the old one

  csrColIndexA = (int *)malloc(nnzA * sizeof(int)); //init
  csrValueA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE)); //init 
  //this will order the (potentially) unordered data that is currently in the temp arrays
  if (isSymmetric)
    {
      for (int i = 0; i < nnzA_mtx; i++)
        {
	  //Not on the diagonal (like y = -x) that the symmetry is defined around
	  if (temp_csrRowIndexA[i] != temp_csrColIndexA[i])
            {
	      int offset = csrRowPtrA[temp_csrRowIndexA[i]] + temp_csrRowPtrA[temp_csrRowIndexA[i]];
	      csrColIndexA[offset] = temp_csrColIndexA[i];
	      csrValueA[offset] = temp_csrValueA[i];
	      temp_csrRowPtrA[temp_csrRowIndexA[i]]++;
	      //Symmetry piece
	      offset = csrRowPtrA[temp_csrColIndexA[i]] + temp_csrRowPtrA[temp_csrColIndexA[i]];
	      csrColIndexA[offset] = temp_csrRowIndexA[i];
	      csrValueA[offset] = temp_csrValueA[i];
	      temp_csrRowPtrA[temp_csrColIndexA[i]]++;
            }
	  else
            {
	      int offset = csrRowPtrA[temp_csrRowIndexA[i]] + temp_csrRowPtrA[temp_csrRowIndexA[i]];
	      csrColIndexA[offset] = temp_csrColIndexA[i];
	      csrValueA[offset] = temp_csrValueA[i];
	      temp_csrRowPtrA[temp_csrRowIndexA[i]]++;
            }
        }
    }
  else
    {
      for (int i = 0; i < nnzA_mtx; i++)
        {
	  //Same block as all above except the symmetry one which is similar but flipped obviously
	  int offset = csrRowPtrA[temp_csrRowIndexA[i]] + temp_csrRowPtrA[temp_csrRowIndexA[i]];
	  csrColIndexA[offset] = temp_csrColIndexA[i];
	  csrValueA[offset] = temp_csrValueA[i];
	  temp_csrRowPtrA[temp_csrRowIndexA[i]]++;
        }
    }

  // free temp space
  free(temp_csrColIndexA);
  free(temp_csrValueA);
  free(temp_csrRowIndexA);
  free(temp_csrRowPtrA);



  //Use the now init CSR



  
  return 0;
}
