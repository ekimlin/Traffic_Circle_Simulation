/*******************************************************************************
  Title          : utilities.c
  Author         : Stewart Weiss
  Created on     : February 10, 2014
  Description    : Various functions used for MPI and non-MPI programs
  Purpose        : 
  Build with     : mpicc -c utilities.c

  Modifications: Added argument FILE *stream to collect_and_print_matrix_byrows() so that user
                 can choose to print to file or standard output. -E.Kimlin 10.11.16
 
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "Sweiss_Utilities.h"

#define PROMPT_MSG         1
#define RESPONSE_MSG       2



/******************************************************************************/
/** get_type() returns a constant representing MPI_Datatype argument
 *  @param    MPI_Datatype  t  
 *  @return   integer constant
 */
int get_type (MPI_Datatype t) 
{
    if ( ( t == MPI_BYTE) ||
         ( t == MPI_CHAR) ||
         ( t == MPI_SIGNED_CHAR) ||
         ( t == MPI_UNSIGNED_CHAR) )
        return CHAR;

    if ( t == MPI_DOUBLE )
        return DOUBLE;
    if (t == MPI_FLOAT) 
        return FLOAT;

    if ( ( t == MPI_INT) ||
         ( t == MPI_UNSIGNED) )
        return INT;

    if ( ( t == MPI_LONG) ||
         ( t == MPI_UNSIGNED_LONG) )
        return LONG;

    if ( ( t == MPI_LONG_LONG) ||
         ( t == MPI_LONG_LONG_INT) ||
         ( t == MPI_UNSIGNED_LONG_LONG) )
        return LLONG;
    
   return -1;
}



/******************************************************************************/
/** get_size() returns the number of bytes in MPI_Datatype argument
 *  @param    MPI_Datatype  t  
 *  @return   number of bytes in t
 */
int get_size (MPI_Datatype t) 
{
    if ( ( t == MPI_BYTE) ||
         ( t == MPI_CHAR) ||
         ( t == MPI_SIGNED_CHAR) ||
         ( t == MPI_UNSIGNED_CHAR) )
        return sizeof(char);

    if ( t == MPI_SHORT )
        return sizeof(short int);

    if ( t == MPI_DOUBLE )
        return sizeof(double);
    if (t == MPI_FLOAT) 
        return sizeof(float);

    if ( ( t == MPI_INT) ||
         ( t == MPI_UNSIGNED) )
        return sizeof(int);

    if ( ( t == MPI_LONG) ||
         ( t == MPI_UNSIGNED_LONG) )
        return sizeof(long int);

    if ( ( t == MPI_LONG_LONG) ||
         ( t == MPI_LONG_LONG_INT) ||
         ( t == MPI_UNSIGNED_LONG_LONG) )
        return sizeof(long long int);
    
   return -1;
}

/******************************************************************************/
/** terminate() prints an error message and terminates calling process 
 *  @param   int id        [IN] rank of calling process
 *  @param   char* message [IN] error message to print
 *  @post    A message is printed on standard output only if id == 0
             but the caller is always terminated with MPI_Finalize.
 */
void terminate (
   int   id,            /* IN - Process rank */
   char *error_message) /* IN - Message to print */
{
   if ( 0 == id ) {
      printf ("%s", error_message);
      fflush (stdout);
   }
   MPI_Finalize();
   exit (0);
}

/******************************************************************************/

/* owner(r,p,n) is the rank of the process that owns element r */
inline int owner( int row, int num_procs, int total_elements )
{
    return ( num_procs * (row+1) -1 ) / total_elements;
}
int size_of_block(int id, int ntotal_elements, int p ) {
    return ( ( ( id + 1) * ntotal_elements ) / p ) - 
           ( ( id *      ntotal_elements ) / p );
}
/******************************************************************************/
void alloc_matrix( 
        int     nrows,          /* number of rows in matrix                   */
        int     ncols,          /* number of columns in matrix                */
        size_t  element_size,   /* number of bytes per matrix element         */
        void  **matrix_storage, /* address of linear storage array for matrix */
        void ***matrix,         /* address of start of matrix                 */
        int    *errvalue)       /* return code for error, if any              */
{
    int   i;
    void *ptr_to_row_in_storage; /* pointer to a place in linear storage array
                                    where a row begins                        */
    void **matrix_row_start;     /* address of a 2D matrix row start pointer
                                    e.g., address of (*matrix)[row]           */
    size_t total_bytes;          /* amount of memory to allocate              */

    //printf("alloc_matrix called with r=%d,c=%d,e=%d\n",nrows, ncols, element_size);

    total_bytes = nrows * ncols * element_size;

    /* Step 1: Allocate an array of nrows * ncols * element_size bytes  */  
    *matrix_storage = malloc(total_bytes);
    if ( NULL == *matrix_storage ) {
        /* malloc failed, so set error code and quit */
        *errvalue = MALLOC_ERROR;
        return;
    }

    memset(*matrix_storage, 0, total_bytes );

    /* Step 2: To create the 2D matrix, first allocate an array of nrows 
       void* pointers */   
    *matrix = malloc (nrows * sizeof(void*));
    if ( NULL == *matrix ) {
        /* malloc failed, so set error code and quit */
        *errvalue = MALLOC_ERROR;
        return;
    }


    /* Step 3: (The hard part) We need to put the addresses into the
       pointers of the 2D matrix that correspond to the starts of rows
       in the linear storage array. The offset of each row in linear storage 
       is a multiple of (ncols * element_size) bytes.  So we initialize
       ptr_to_row_in_storage to the start of the linear storage array and
       add (ncols * element_size) for each new row start.
       The pointers in the array of pointers to rows are of type void* 
       so an increment operation on one of them advances it to the next pointer.
       Therefore, we can initialize matrix_row_start to the start of the 
       array of pointers, and auto-increment it to advance it.
    */

    /* Get address of start of array of pointers to linear storage, 
       which is the address of first pointer, (*matrix)[0]   */
    matrix_row_start = (void*) &(*matrix[0]);

    /* Get address of start of linear storage array */
    ptr_to_row_in_storage = (void*) *matrix_storage;

    /* For each matrix pointer, *matrix[i], i = 0... nrows-1, 
       set it to the start of the ith row in linear storage */
    for ( i = 0; i < nrows; i++ ) {
        /* matrix_row_start is the address of (*matrix)[i] and
           ptr_to_row_in_storage is the address of the start of the 
           ith row in linear storage.
           Therefore, the following assignment changes the contents of 
           (*matrix)[i]  to store the start of the ith row in linear storage 
        */
        *matrix_row_start = (void*) ptr_to_row_in_storage;

        /* advance both pointers */
        matrix_row_start++;     /* next pointer in 2d array */
        ptr_to_row_in_storage +=  ncols * element_size; /* next row */
    }
    *errvalue = SUCCESS;
}

void read_and_distribute_2dblock_matrix (
          char        *filename,       /* [IN]  name of file to read          */
          void      ***matrix,         /* [OUT] matrix to fill with data      */
          void       **matrix_storage, /* [OUT] linear storage for the matrix */
          MPI_Datatype dtype,          /* [IN]  matrix element type           */
          int         *nrows,          /* [OUT] number of rows in matrix      */
          int         *ncols,          /* [OUT] number of columns in matrix   */
          int         *errval,         /* [OUT] sucess/error code on return   */
          MPI_Comm     cart_comm)      /* [IN]  communicator handle           */
{
    int    i,j,k;           /* various loop index variables                  */
    int    grid_id;         /* process rank in the cartesian grid            */
    int    p;               /* number of processes in the cartesian grid     */
    size_t element_size;    /* number of bytes in matrix element type        */
    int    mpi_initialized; /* flag to check if MPI_Init was called already  */
    FILE   *file;           /* input file stream pointer                     */
    int    nlocal_rows;     /* number of rows that calling process "owns"    */
    int    nlocal_cols;     /* number of columns that calling process "owns" */
    MPI_Status   status;    /* result of MPI_Recv call                       */
    int    dest_id;         /* rank of receiving process in cartesian grid   */
    int    grid_coord[2];   /* process coordinates in the grid               */
    int    grid_periodic[2];/* flags indicating if grid wraps around         */
    int    grid_size[2];    /* dimensions of grid                            */
    void*  buffer;          /* address of temp location to store rows        */
    int    block_coord[2];  /* coordinates in grid of current block          */
    void*  source_address;  /* address of block to be sent                   */
    void*  dest_address;    /* location where block is to be received        */
    
    /* Make sure we are being called by a program that init-ed MPI */
    MPI_Initialized(&mpi_initialized);
    if ( !mpi_initialized ) {
       *errval = -1;
       return;
    }
  
    /* Get process rank in grid and the number of processes in group */
    MPI_Comm_rank (cart_comm, &grid_id);  
    MPI_Comm_size (cart_comm, &p);            
   
    /* Get the number of bytes in a matrix element */
    element_size = get_size (dtype);
    if ( element_size <= 0 ) {
       *errval = -1;
       return;
    }
       
    /* Process 0 opens the file and reads the number of rows and columns */
    if ( 0 == grid_id ) { 
        /* Process 0 opens the binary file containing the matrix and 
           reads the first two numbers, which are the number of rows and 
           columns respectively. */
        file = fopen (filename, "r");
        if ( NULL == file ) {
            *nrows = 0;
            *ncols = 0;
        }
        else { /* successful open */
            fread (nrows, sizeof(int), 1, file);
            fread (ncols, sizeof(int), 1, file);
        }      
    }

    /* Process 0 broadcasts the numbers of rows to all other processes. */
    MPI_Bcast (nrows, 1, MPI_INT, 0, cart_comm);

    /* All processes check value of *nrows; if 0 it indicates failed open */
    if ( 0 == *nrows  ) {
       *errval = -1;
       return;
    }

    /* Process 0 broadcasts the numbers of columns to all other processes. 
       No need to check whether *ncols is zero. */
    MPI_Bcast (ncols, 1, MPI_INT, 0, cart_comm);

    /* All processes obtain the grid's topology so they can determine
       their block sizes. */
    MPI_Cart_get (cart_comm, 2, grid_size, grid_periodic, grid_coord);

    /* Each process sets nlocal_rows = the number of rows the process owns and
       local_cols to the number of columns it owns.
       The number of rows depends on the process's row coordinate, *nrows, 
       and the number of grid rows in total. This implements the formula
             blocks = floor((i+1)*n/p) - floor(i*n/p)
       where i is grid coordinate in given dimension, n is either total
       number of rows, or total number of columns, and p is the number of
       processes in grid in the given dimension.
    */
    nlocal_rows = size_of_block( grid_coord[0], *nrows, grid_size[0] );
    nlocal_cols = size_of_block( grid_coord[1], *ncols, grid_size[1] );

    /* Each process creates its linear storage and 2D matrix for accessing 
       the elements of its assigned rows. It needs storage for a 2D matrix
       of nlocal_rows by nlocal_cols elements. */
    alloc_matrix( nlocal_rows, nlocal_cols, element_size,
                  matrix_storage, 
                  matrix,         
                  errval);
    if ( SUCCESS != *errval ) {
         MPI_Abort (cart_comm, *errval);
    }

   /* Grid process 0 reads in the matrix one row at a time
      and distributes each row among the MPI processes. The first step is
      to allocate storage for one row of the matrix. */
    if ( 0 == grid_id ) {
        buffer = malloc (*ncols * element_size);
        if ( buffer == NULL ) { 
            MPI_Abort (cart_comm, *errval);
        }
    }

    /* This is the read and distribute loop. Process 0 will read a row
       and send it to the processes that are supposed to have it. It needs to
       break it into blocks, with successive blocks going to the processes
       in the same grid row but successive grid columns. */

    for (i = 0; i < grid_size[0]; i++) { /* for each grid row */
        /* Set block_coord[0] to the current grid row index */
        block_coord[0] = i;

        /* For every matrix row that is part of this grid row */
        for (j = 0; j < size_of_block(i, *nrows, grid_size[0] ); j++) {

            /* Process 0 reads  a row of the matrix */
            if ( 0 == grid_id ) {
                fread (buffer, element_size, *ncols, file);
            }

            /* Every process executes this loop. For each grid column within
               the current grid row ... */
            for (k = 0; k < grid_size[1]; k++) {
                block_coord[1] = k;

                /* Determine the grid id of the process in grid position
                   [i,k].  This is the destination process. Its id is returned 
                   in dest_id. */
                MPI_Cart_rank (cart_comm, block_coord, &dest_id);

                /* Process 0 needs to determine the start of the block to
                   be sent to process dest_id. This is the start in a row
                   with *ncols elements assigned to kth process out of
                   grid_size[1] many processes in that row. */
                if ( 0 == grid_id ) {
                    source_address = buffer + 
                                 ( (k*(*ncols))/grid_size[1] ) * element_size;
                    /* The process has to make sure it does not try to send to
                       itself. If so it does a memory copy instead. */
                    if (0 == dest_id ) {
                        /* It is sending to itself */
                        dest_address = (*matrix)[j];
                        memcpy (dest_address, source_address, 
                               nlocal_cols * element_size);                  
                    } 
                    else {
                        /* It is sending to another process */
                        int blocksize = size_of_block(k,*ncols, grid_size[1]);
                        MPI_Send (source_address,blocksize, dtype,
                                  dest_id, 0, cart_comm);
                    }
                }
                else if (grid_id == dest_id) {
                         MPI_Recv ((*matrix)[j], nlocal_cols, dtype, 0,
                          0, cart_comm, &status);
                }
            } /* end for k */
        } /* end for j */
    } /* for i */

    if (grid_id == 0) 
        free (buffer);
    *errval = 0;
}

/******************************************************************************/
void collect_and_print_2dblock_matrix (
   void       **a,            /* IN -2D matrix */
   MPI_Datatype dtype,        /* IN -Matrix element type */
   int          m,            /* IN -Matrix rows */
   int          n,            /* IN -Matrix columns */
   FILE         *stream,      /* OUT-stream on which to print */
   MPI_Comm     grid_comm)    /* IN - Communicator */
{
   void      *buffer;         /* Room to hold 1 matrix row */
   int        coords[2];      /* Grid coords of process ending elements */
   int        element_size;    /* Bytes per matrix element */
   int        els;            /* Elements received */
   int        grid_coords[2]; /* Coords of this process */
   int        grid_id;        /* Process rank in grid */
   int        grid_period[2]; /* Wraparound */
   int        grid_size[2];   /* Dims of process grid */
   int        i, j, k;
   void      *laddr;          /* Where to put subrow */
   int        local_cols;     /* Matrix cols on this proc */
   int        p;              /* Number of processes */
   int        src;            /* ID of proc with subrow */
   MPI_Status status;         /* Result of receive */

   MPI_Comm_rank (grid_comm, &grid_id);
   MPI_Comm_size (grid_comm, &p);
   element_size = get_size (dtype);

   MPI_Cart_get (grid_comm, 2, grid_size, grid_period,
      grid_coords);
   local_cols = size_of_block(grid_coords[1], n, grid_size[1]);

   if (0 == grid_id)
      buffer = malloc ( n * element_size);

   /* For each row of the process grid */
   for (i = 0; i < grid_size[0]; i++) {
      coords[0] = i;

      /* For each matrix row controlled by the process row */
      for (j = 0; j < size_of_block(i,m, grid_size[0] ); j++) {

         /* Collect the matrix row on grid process 0 and
            print it */
         if (0 == grid_id) {
            for (k = 0; k < grid_size[1]; k++) {
               coords[1] = k;
               MPI_Cart_rank (grid_comm, coords, &src);
               els = size_of_block(k,n, grid_size[1]);
               laddr = buffer +
                  ((k*n)/grid_size[1]) * element_size;
               if (src == 0) {
                  memcpy (laddr, a[j], els * element_size);
               } else {
                  MPI_Recv(laddr, els, dtype, src, 0,
                     grid_comm, &status);
               }
            }
            print_vector (buffer, n, dtype, stream);
            fprintf (stream, "\n");
         } 
         else if (grid_coords[0] == i) {
            MPI_Send (a[j], local_cols, dtype, 0, 0,
               grid_comm);
         }
      }
   }
   if (0 == grid_id) {
      free (buffer);
      fprintf (stream, "\n");
   }
}


/******************************************************************************/
void print_vector (
        void   *vector,         /* vector to be printed        */
        int     n,              /* number of elements in vector*/
        MPI_Datatype dtype,     /* MPI type                    */
        FILE  *stream)          /* stream on which to print    */
{
    int i;
    int etype = get_type(dtype);
   
    for (i = 0; i < n; i++) {
            switch (etype) {
            case DOUBLE:
                fprintf (stream, "%.4f ", ((double *)vector)[i]); break;
            case FLOAT:
                fprintf (stream,"%.4f ", ((float *)vector)[i]);   break;
            case INT:
                fprintf (stream,"%d ", ((int *)vector)[i]);       break;
            case CHAR:
                fprintf (stream,"%6c ", ((char *)vector)[i]);      break;
            case LONG:
                fprintf (stream,"%6ld ", ((long int *)vector)[i]); break;
            case LLONG:
                fprintf (stream,"%6lld ", ((long long int *)vector)[i]);
            }
    }
}

void print_full_vector (
    void        *v,      /* IN - Address of vector */
    int          n,      /* IN - Elements in vector */
    MPI_Datatype dtype,  /* IN - Vector element type */
    MPI_Comm     comm)   /* IN - Communicator */
{
    int id;              /* Process rank */

    MPI_Comm_rank (comm, &id);
   
    if (0 == id) {
        print_vector (v, n, dtype, stdout);
        printf("\n");
    }
}

void print_block_vector (
    void        *v,       /* IN - Address of vector */
    MPI_Datatype dtype,   /* IN - Vector element type */
    int          n,       /* IN - Elements in vector */
    MPI_Comm     comm)    /* IN - Communicator */
{
    int        datum_size; /* Bytes per vector element */
    int        i;
    int        prompt;     /* Dummy variable */
    MPI_Status status;     /* Result of receive */
    void       *tmp;       /* Other process's subvector */
    int        id;         /* Process rank */
    int        p;          /* Number of processes */

    MPI_Comm_size (comm, &p);
    MPI_Comm_rank (comm, &id);
    datum_size = get_size (dtype);

    if ( 0 == id ) {
        print_vector (v,  size_of_block(id,n,p), dtype, stdout);
        if (p > 1) {
            tmp = malloc (size_of_block(p-1,n,p)*datum_size);
            for (i = 1; i < p; i++) {
                MPI_Send (&prompt, 1, MPI_INT, i, PROMPT_MSG, comm);
                MPI_Recv (tmp, size_of_block(i,n,p), dtype, i,
                          RESPONSE_MSG, comm, &status);
                print_vector (tmp,  size_of_block(i,n,p), dtype, stdout);
            }
            free (tmp);
        }
        printf ("\n\n");
    } 
    else {
        MPI_Recv (&prompt, 1, MPI_INT, 0, PROMPT_MSG, comm, &status);
        MPI_Send (v, size_of_block(id,n,p), dtype, 0,
                  RESPONSE_MSG, comm);
    }
}


/******************************************************************************/
/** init_communication_arrays() initialize arrays to pass to MPI gather/scatterv
 *  @param  int p            [IN]   Number of processes 
 *  @param  int n            [IN]   Total number of elements
 *  @param  int *count       [OUT]  Array of counts
 *  @return int *offset      [OUT]  Array of displacements 
 */
void init_communication_arrays (
    int p,          /* IN - Number of processes */
    int n,          /* IN - Total number of elements */
    int *count,     /* OUT - Array of counts */
    int *offset)    /* OUT - Array of displacements */
{
    int i;

    count[0]  = size_of_block(0,n,p);
    offset[0] = 0;
    for (i = 1; i < p; i++) {
        offset[i] = offset[i-1] + count[i-1];
        count[i]  = size_of_block(i,n,p);
    }
}

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
        )
{
    int *recv_count;  /* Elements contributed by each process */
    int *recv_offset; /* Displacement in concatenated array */
    int id;           /* Process id */
    int p;            /* Processes in communicator */

    MPI_Comm_size (comm, &p);
    MPI_Comm_rank (comm, &id);

    /* Allocate count and offset arrays and bail out if either fails. */
    recv_count  = malloc ( p * sizeof(int));
    recv_offset = malloc ( p * sizeof(int));
    if ( NULL == recv_offset || NULL == recv_count ) {
        printf ("malloc failed for process %d\n", id);
        MPI_Abort (MPI_COMM_WORLD, MALLOC_ERROR);
    }

    /* Fill the count and offset arrays to pass to MPI_Allgatherv
       so that the blocks are concatenated by process rank in the output
       vector */
    init_communication_arrays (p, n, recv_count, recv_offset);

    /* Use MPI_Allgatherv to copy the distributed blocks from invec in
       each process into a replicated outvec in each process. */
    MPI_Allgatherv (invec, recv_count[id], dtype, outvec, recv_count,
                    recv_offset, dtype, comm);

    /* Release the storage for the count and offset arrays. */
    free (recv_count);
    free (recv_offset);
}


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
char*  init_random( int state_size )
{
    char * state;
    state  = (char*) malloc ( state_size * sizeof(char));
    if ( NULL != state )
        initstate(time(NULL), state, state_size);
    return state;
}

void   finalize ( char* state )
{
    free (state );
}

/** uniform_random()  returns a uniformly distributed random number in [0,1]
 *  @return double  a pointer to the state array allocated for random()
 *  @pre           Either init_random() should have been called or srandom()
 */
double uniform_random()
{
    return (double) (random()) / RAND_MAX; 
}