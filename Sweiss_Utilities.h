#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <mpi.h>

#define SUCCESS             0
#define MALLOC_ERROR        1
#define OPEN_FILE_ERROR     2
#define TYPE_ERROR          3
#define FILE_READ_ERROR     4

#define MIN(a,b)           ((a)<(b)?(a):(b))
#define MAX(a,b)           ((a)>(b)?(a):(b))

#define  FLOAT  0
#define  DOUBLE 1
#define  INT    2
#define  CHAR   3
#define  LONG   4
#define  LLONG  5


/******************************************************************************/
/** get_type() returns a constant representing MPI_Datatype argument
 *  @param    MPI_Datatype  t  
 *  @return   integer constant
 */
int get_type (MPI_Datatype t);

/******************************************************************************/
/** get_size() returns the number of bytes in MPI_Datatype argument
 *  @param    MPI_Datatype  t  [IN]  type of element
 *  @return   number of bytes in t
 */
int get_size (MPI_Datatype t);


/******************************************************************************/
/** size_of_block()  returns number of elements in block of array
 *  @comment  This assumes that elements of an array have been distributed
 *            among processes so that the first location is 
 *            id*n/p and last is (id+1)*n/p -1.
 *  @param    int id              [IN] rank of calling process 
 *  @param    int ntotal_elements [IN] number of elements in array
 *  @param    int p               [IN] number of processes in group
 *  @return   number of elements assigned to process id 
 */
 /*
inline  int size_of_block(int id, int ntotal_elements, int p ) {
    return ( ( ( id + 1) * ntotal_elements ) / p ) - 
           ( ( id *      ntotal_elements ) / p );
} */
int size_of_block(int id, int ntotal_elements, int p );
/******************************************************************************/
/** owner() returns the rank of the process that owns a given element
 *  @comment  This assumes that elements of an array have been distributed
 *            among processes so that the first location is 
 *            id*n/p and last is (id+1)*n/p -1.
 *  @param  int row          [IN]  index of element 
 *  @param  int num_procs    [IN]  number of processes in group
 *  @param  int num_elements [IN]  number of elements in array
 *  @return int rank of process that own element.
 */
inline int owner( 
        int row, 
        int num_procs, 
        int num_elements );

/******************************************************************************/
/** terminate() prints an error message and terminates calling process 
 *  @param   int id        [IN] rank of calling process
 *  @param   char* message [IN] error message to print
 *  @post    A message is printed on standard output only if id == 0
             but the caller is always terminated with MPI_Finalize.
 */
void terminate (
        int   id,            /* IN - Process rank */
        char *error_message) ;/* IN - Message to print */

/******************************************************************************/
/** alloc_matrix(r,c,e, &Mstorage, &M, &err)
 *  If &err is SUCCESS, on return it allocated storage for two arrays in 
 *  the heap. Mstorage is a linear array large enough to hold the elements of
 *  an r by c 2D matrix whose elements are e bytes long. The other, M, is a 2D
 *  matrix such that M[i][j] is the element in row i and column j.
 */
void alloc_matrix( 
        int     nrows,          /* number of rows in matrix                   */
        int     ncols,          /* number of columns in matrix                */
        size_t  element_size,   /* number of bytes per matrix element         */
        void  **matrix_storage, /* address of linear storage array for matrix */
        void ***matrix,         /* address of start of matrix                 */
        int    *errvalue        /* return code for error, if any              */
        );

void read_and_distribute_2dblock_matrix (
          char        *filename,       /* [IN]  name of file to read          */
          void      ***matrix,         /* [OUT] matrix to fill with data      */
          void       **matrix_storage, /* [OUT] linear storage for the matrix */
          MPI_Datatype dtype,          /* [IN]  matrix element type           */
          int         *nrows,          /* [OUT] number of rows in matrix      */
          int         *ncols,          /* [OUT] number of columns in matrix   */
          int         *errval,         /* [OUT] success/error code on return  */
          MPI_Comm     cart_comm);     /* [IN]  communicator handle           */
/******************************************************************************/
void collect_and_print_2dblock_matrix  (
         void       **a,            /* IN -2D matrix */
         MPI_Datatype dtype,        /* IN -Matrix element type */
         int          m,            /* IN -Matrix rows */
         int          n,            /* IN -Matrix columns */
         FILE       *file,          /* OUT-stream on which to print */
         MPI_Comm     grid_comm);    /* IN - Communicator */

/******************************************************************************/
/** print_vector() prints a vector on a line of the given file stream
 *
 *
 */
void print_vector (
        void   *vector,         /* vector to be printed        */
        int     n,              /* number of elements in vector*/
        MPI_Datatype dtype,     /* MPI type                    */
        FILE  *stream);         /* stream on which to print    */

/******************************************************************************/
/** print_full_vector() prints a vector on a line of stdout
 *  
 *
 */
void print_full_vector (
    void        *v,      /* IN - Address of vector */
    int          n,      /* IN - Elements in vector */
    MPI_Datatype dtype,  /* IN - Vector element type */
    MPI_Comm     comm);   /* IN - Communicator */


/******************************************************************************/
/** print_block_vector() prints a block decomposed vector on a line of stdout
 * 
 *
 */
void print_block_vector (
   void        *v,       /* IN - Address of vector */
   MPI_Datatype dtype,   /* IN - Vector element type */
   int          n,       /* IN - Elements in vector */
   MPI_Comm     comm) ;   /* IN - Communicator */

/******************************************************************************/
/** init_communication_arrays() initialize arrays to pass to MPI gather/scatterv
 *  @param  int p            [IN]   Number of processes 
 *  @param  int n            [IN]   Total number of elements
 *  @param  int *count       [OUT]  Array of counts
 *  @return int *offset      [OUT]  Array of displacements 
 */
void init_communication_arrays (
        int p,                 
        int n,                
        int *count,            
        int *offset           
        );

/******************************************************************************/
/** replicate_block_vector() copies a distributed vector into every process
 *  @param  void        *invec   [IN]   Block-distributed vector
 *  @param  int          n       [IN]   Total number of elements in vector
 *  @param  MPI_Datatype dtype   [IN]   MPI element type
 *  @param  void        *outvec  [OUT]  Replicated vector
 *  @param  MPI_Comm     comm    [IN]  Communicator
 */
void replicate_block_vector (
        void        *invec,  
        int          n,      
        MPI_Datatype dtype,  
        void        *outvec,   
        MPI_Comm     comm   
        );

/*******************************************************************************
                           Random Number Routines
*******************************************************************************/


/******************************************************************************/
/** init_random()  initializes the state for the C random() function
 *  @param  int    state_size  [IN]  Size of state array for random to use
 *  @return char*  a pointer to the state array allocated for random()
 *  @post          After this call, an array of size state_size*sizeof(char) has
 *                 been allocated and initialized by C initstate(). It must be
 *                 freed by calling free()
 */
char*  init_random( int state_size );


void   finalize ( char* state );

/** uniform_random()  returns a uniformly distributed random number in [0,1]
 *  @return double  a pointer to the state array allocated for random()
 *  @pre           Either init_random() should have been called or srandom()
 */
double uniform_random();


#endif /* __UTILITIES_H__ */