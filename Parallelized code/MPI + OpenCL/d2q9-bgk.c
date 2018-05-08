/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook rebound_collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

#define OCL_KERNELS_FILE "kernel.cl"

#define MASTER 0

/* struct to hold the parameter values */
typedef struct
{
    int nx;           /* no. of cells in x-direction */
    int ny;           /* no. of cells in y-direction */
    int maxIters;     /* no. of iterations */
    int reynolds_dim; /* dimension for Reynolds number */
    float density;    /* density per link */
    float accel;      /* density redistribution */
    float omega;      /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct //__attribute__ ((aligned))
{
    float speeds[NSPEEDS];
} t_speed;

/* struct to hold OpenCL objects */
typedef struct
{
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program;

    cl_kernel accelerate_flow;
    cl_kernel propagate;
    //cl_kernel rebound;
    cl_kernel rebound_collision;
    cl_kernel av_velocity;

    cl_mem d_obstacles;
    cl_mem d_cells;
    cl_mem d_tmp_cells;
    cl_mem d_groupsums;
    
    size_t av_work_group_size;
    int wgsize;
} t_ocl;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & rebound_collision()
*/
int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles);
int accelerate_flow(const t_param params, t_speed *cells, int *obstacles);
int propagate(const t_param params, t_speed *cells, t_speed *tmp_cells);
int rebound(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles);
int rebound_collision(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles);
int write_values(const t_param params, t_speed *cells, int *obstacles, float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed *cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed *cells, int *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed *cells, int *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

// --- OpenCL modules below ----
void checkError(cl_int err, const char *op, const int line);
cl_device_id selectOpenCLDevice();
void OCL_initialise(t_ocl *ocl, const int N);
void OCL_finalise(t_ocl ocl);
void OCL_setKernelArgs();
// --- end of OpenCL modules ---

/*
** main program:
** initialise, timestep loop, finalise
*/

int MRank, MSize;
MPI_Datatype TMPI_T_SPEED;
int ll, rr;
int nonblocked_cells = 0;
int num_cells;
t_ocl ocl;
cl_int err;
int main(int argc, char *argv[])
{
    char *paramfile = NULL;    /* name of the input parameter file */
    char *obstaclefile = NULL; /* name of a the input obstacle file */
    t_param params;            /* struct to hold parameter values */
    t_speed *cells = NULL;     /* grid containing fluid densities */

    t_speed *tmp_cells = NULL; /* scratch space */
    int *obstacles = NULL;     /* grid indicating which cells are blocked */
    float *av_vels = NULL;     /* a record of the av. velocity computed for each timestep */

    struct timeval timstr; /* structure to hold elapsed time */
    struct rusage ru;      /* structure to hold CPU time--system and user */
    double tic, toc;       /* floating point numbers to calculate elapsed wallclock time */
    double usrtim;         /* floating point number to record elapsed user CPU time */
    double systim;         /* floating point number to record elapsed system CPU time */

    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    //----- MPI initialise & load data  --------------

    MPI_Init(NULL, NULL);

    /* Get rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &MRank);
    MPI_Comm_size(MPI_COMM_WORLD, &MSize);

    MPI_Type_contiguous(NSPEEDS, MPI_FLOAT, &TMPI_T_SPEED);
    MPI_Type_commit(&TMPI_T_SPEED);
    //------------------------------------------------

    /* initialise our data structures and load values from file */
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels); // MPI changed

    MPI_Bcast(&(params.nx), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&(params.ny), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&(params.maxIters), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&(params.reynolds_dim), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&(params.density), 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&(params.accel), 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&(params.omega), 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    MPI_Bcast(&nonblocked_cells, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    num_cells = params.nx * params.ny;

    MPI_Bcast(obstacles, num_cells, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(cells, num_cells, TMPI_T_SPEED, MASTER, MPI_COMM_WORLD);

    ll = params.ny / MSize * (MRank);
    rr = (MRank == MSize - 1) ? params.ny : params.ny / MSize * (MRank + 1);

    MPI_Barrier(MPI_COMM_WORLD);

    //printf("%d\n",sizeof(t_speed));
    //printf("%d %d %d %d \n",MRank,params.ny,ll,rr);

    //----End of MPI Init----------------------

    //----Beginning of OCL Init----------------

    OCL_initialise(&ocl, num_cells);

    // Write data to OpenCL buffer on the device
    err = clEnqueueWriteBuffer(ocl.queue, ocl.d_obstacles, CL_TRUE, 0, sizeof(int) * num_cells,
                               obstacles, 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);

    err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells, CL_TRUE, 0, sizeof(t_speed) * num_cells,
                               cells, 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);

    err = clEnqueueWriteBuffer(ocl.queue, ocl.d_tmp_cells, CL_TRUE, 0, sizeof(t_speed) * num_cells,
                               tmp_cells, 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);

    OCL_setKernelArgs(params);
    //----End of OCL Init----------------------
    
    if (MRank == MASTER)
    {
        gettimeofday(&timstr, NULL);
        tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    }

    /* iterate for maxIters timesteps */
    for (int tt = 0; tt < params.maxIters; tt++)
    {
        timestep(params, cells, tmp_cells, obstacles);
        float local_av = av_velocity(params, cells, obstacles);
        float global_av;
        MPI_Reduce(&local_av, &global_av, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        if (MRank == MASTER)
        {
            av_vels[tt] = global_av;
        }
    }
    // ------------------
    // load cells from device
    err = clEnqueueReadBuffer(ocl.queue, ocl.d_cells, CL_TRUE, sizeof(t_speed) * params.nx * ll, sizeof(t_speed) * params.nx * (rr - ll), &cells[params.nx * ll], 0, NULL, NULL);

    if (MRank != MASTER)
    {
        MPI_Ssend(&(cells[ll * params.nx]), (rr - ll) * params.nx, TMPI_T_SPEED, MASTER, 3, MPI_COMM_WORLD);
    }
    else
    {
        for (int i = 1; i < MSize; ++i)
        {
            int llx = params.ny / MSize * i;
            int rrx = (i == MSize - 1) ? params.ny : params.ny / MSize * (i + 1);
            MPI_Recv(&(cells[llx * params.nx]), (rrx - llx) * params.nx, TMPI_T_SPEED, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    // ------------------

    if (MRank == MASTER)
    {
        gettimeofday(&timstr, NULL);
        toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
        getrusage(RUSAGE_SELF, &ru);
        timstr = ru.ru_utime;
        usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
        timstr = ru.ru_stime;
        systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

        /* write final values and free memory */
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
        printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
        printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
        printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
        write_values(params, cells, obstacles, av_vels);
        finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
    }
    

    //-----------------------
    OCL_finalise(ocl);
    MPI_Finalize();
    //-----------------------
    return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles)
{
    accelerate_flow(params, cells, obstacles);
    propagate(params, cells, tmp_cells);
    //rebound(params, cells, tmp_cells, obstacles);
    rebound_collision(params, cells, tmp_cells, obstacles);
    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed *cells, int *obstacles)
{

    if (MRank == MSize - 1)
    {
        // Enqueue kernel
        size_t global[1] = {params.nx};
        err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow, 1, NULL, global, NULL,
                                     0, NULL, NULL);
        //checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

        // Wait for kernel to finish
        err = clFinish(ocl.queue);
        //checkError(err, "waiting for accelerate_flow kernel", __LINE__);

    }
    return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed *cells, t_speed *tmp_cells)
{
    /* loop over _all_ cells */

    int up_rank = (MRank + 1) >= MSize ? 0 : MRank + 1;
    int down_rank = (MRank - 1) < 0 ? (MSize - 1) : (MRank - 1);
    int up_cell = rr >= params.ny ? 0 : rr;
    int down_cell = (ll - 1) < 0 ? (params.ny - 1) : (ll - 1);

    err = clEnqueueReadBuffer(ocl.queue, ocl.d_cells, CL_TRUE, sizeof(t_speed) * params.nx * (rr - 1), sizeof(t_speed) * params.nx,
                               &cells[params.nx * (rr - 1)], 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);
    // update cells
    err = clEnqueueReadBuffer(ocl.queue, ocl.d_cells, CL_TRUE, sizeof(t_speed) * params.nx * ll, sizeof(t_speed) * params.nx,
                               &cells[params.nx * ll], 0, NULL, NULL);

    MPI_Sendrecv(&(cells[(rr - 1) * params.nx]), params.nx, TMPI_T_SPEED, up_rank, 0, &(cells[down_cell * params.nx]),
                 params.nx, TMPI_T_SPEED, down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&(cells[ll * params.nx]), params.nx, TMPI_T_SPEED, down_rank, 1, &(cells[up_cell * params.nx]),
                 params.nx, TMPI_T_SPEED, up_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // update cells
    err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells, CL_TRUE, sizeof(t_speed) * params.nx * down_cell, sizeof(t_speed) * params.nx,
                               &cells[params.nx * down_cell], 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);
    // update cells
    err = clEnqueueWriteBuffer(ocl.queue, ocl.d_cells, CL_TRUE, sizeof(t_speed) * params.nx * up_cell, sizeof(t_speed) * params.nx,
                               &cells[params.nx * up_cell], 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);



    size_t global[2] = {params.nx, rr - ll};
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.propagate, 2, NULL, global, NULL, 0, NULL, NULL);
    //checkError(err, "enqueueing propagate kernel", __LINE__);

    // Wait for kernel to finish
    err = clFinish(ocl.queue);
    //checkError(err, "waiting for propagate kernel", __LINE__);

    return EXIT_SUCCESS;
}

// int rebound(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles)
// {


//     // size_t global[2] = {params.nx, rr - ll};
//     // err = clEnqueueNDRangeKernel(ocl.queue, ocl.rebound, 2, NULL, global, NULL, 0, NULL, NULL);
//     // //checkError(err, "enqueueing rebound kernel", __LINE__);

//     // // Wait for kernel to finish
//     // err = clFinish(ocl.queue);
//     // //checkError(err, "waiting for rebound kernel", __LINE__);

//     return EXIT_SUCCESS;
// }

int rebound_collision(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles)
{


    size_t global[2] = {params.nx, rr - ll};
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.rebound_collision, 2, NULL, global, NULL, 0, NULL, NULL);
    //checkError(err, "enqueueing rebound_collision kernel", __LINE__);

    // Wait for kernel to finish
    err = clFinish(ocl.queue);
    //checkError(err, "waiting for rebound_collision kernel", __LINE__);

    return EXIT_SUCCESS;
}

float *result;
float av_velocity(const t_param params, t_speed *cells, int *obstacles)
{
    float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

    size_t global[1] = {params.nx * (rr - ll)} , local[1] = {ocl.av_work_group_size};
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.av_velocity, 1, NULL, global, local, 0, NULL, NULL);
    //checkError(err, "enqueueing av_velocity kernel", __LINE__);

    // Wait for kernel to finish
    err = clFinish(ocl.queue);
    //checkError(err, "waiting for av_velocity kernel", __LINE__);

    int ngroups=(int)(params.nx * (rr - ll) / ocl.av_work_group_size);

    //printf("%d\n",ngroups);
    err = clEnqueueReadBuffer(ocl.queue, ocl.d_groupsums, CL_TRUE, 0, sizeof(float) * ngroups, result, 0, NULL, NULL);
    //checkError(err, "writing h_a data", __LINE__);

    for (int i=0;i<ngroups;++i) tot_u+=result[i];



    return tot_u / (float)nonblocked_cells;
}

int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr)
{
    char message[1024]; /* message buffer */
    FILE *fp;           /* file pointer */
    int xx, yy;         /* generic array indices */
    int blocked;        /* indicates whether a cell is blocked by an obstacle */
    int retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1)
        die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1)
        die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1)
        die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1)
        die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->density));

    if (retval != 1)
        die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->accel));

    if (retval != 1)
        die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->omega));

    if (retval != 1)
        die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

    /* main grid */
    *cells_ptr = (t_speed *)malloc(sizeof(t_speed) * (params->ny * params->nx));

    if (*cells_ptr == NULL)
        die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed *)malloc(sizeof(t_speed) * (params->ny * params->nx));

    if (*tmp_cells_ptr == NULL)
        die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

    if (*obstacles_ptr == NULL)
        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /// ---------------------
    if (MRank != MASTER)
    {
        // *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
        return EXIT_SUCCESS;
    }
    /// ---------------------

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density / 9.f;
    float w2 = params->density / 36.f;

    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            /* centre */
            (*cells_ptr)[ii + jj * params->nx].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii + jj * params->nx].speeds[1] = w1;
            (*cells_ptr)[ii + jj * params->nx].speeds[2] = w1;
            (*cells_ptr)[ii + jj * params->nx].speeds[3] = w1;
            (*cells_ptr)[ii + jj * params->nx].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii + jj * params->nx].speeds[5] = w2;
            (*cells_ptr)[ii + jj * params->nx].speeds[6] = w2;
            (*cells_ptr)[ii + jj * params->nx].speeds[7] = w2;
            (*cells_ptr)[ii + jj * params->nx].speeds[8] = w2;
        }
    }

    /* first set all cells in obstacle array to zero */
    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            (*obstacles_ptr)[ii + jj * params->nx] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3)
            die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1)
            die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1)
            die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1)
            die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        (*obstacles_ptr)[xx + yy * params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);

    // ----------------------------------------------
    for (int jj = 0; jj < params->ny; jj++)
        for (int ii = 0; ii < params->nx; ii++)
            if (!(*obstacles_ptr)[ii + jj * params->nx])
                ++nonblocked_cells;
    // ----------------------------------------------
    /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
    *av_vels_ptr = (float *)malloc(sizeof(float) * params->maxIters);

    return EXIT_SUCCESS;
}

int finalise(const t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr)
{
    /*
  ** free up allocated memory
  */
    free(*cells_ptr);
    *cells_ptr = NULL;

    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;

    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed *cells, int *obstacles)
{
    const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
    //==
    ll = 0, rr = params.ny;
    //==
    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed *cells)
{
    float total = 0.f; /* accumulator */

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            for (int kk = 0; kk < NSPEEDS; kk++)
            {
                total += cells[ii + jj * params.nx].speeds[kk];
            }
        }
    }

    return total;
}

int write_values(const t_param params, t_speed *cells, int *obstacles, float *av_vels)
{
    FILE *fp;                     /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density;          /* per grid cell sum of densities */
    float pressure;               /* fluid pressure in grid cell */
    float u_x;                    /* x-component of velocity in grid cell */
    float u_y;                    /* y-component of velocity in grid cell */
    float u;                      /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int idx = ii + jj * params.nx;
            /* an occupied cell */
            if (obstacles[idx])
            {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += cells[idx].speeds[kk];
                }

                /* compute x velocity component */
                u_x = (cells[idx].speeds[1] + cells[idx].speeds[5] + cells[idx].speeds[8] - (cells[idx].speeds[3] + cells[idx].speeds[6] + cells[idx].speeds[7])) / local_density;
                /* compute y velocity component */
                u_y = (cells[idx].speeds[2] + cells[idx].speeds[5] + cells[idx].speeds[6] - (cells[idx].speeds[4] + cells[idx].speeds[7] + cells[idx].speeds[8])) / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
        }
    }

    fclose(fp);

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}

void checkError(cl_int err, const char *op, const int line)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}
cl_device_id selectOpenCLDevice()
{
    cl_int err;
    cl_uint num_platforms = 0;
    cl_uint total_devices = 0;
    cl_platform_id platforms[8];
    cl_device_id devices[MAX_DEVICES];
    char name[MAX_DEVICE_NAME];

    // Get list of platforms

    err = clGetPlatformIDs(8, platforms, &num_platforms);
    //checkError(err, "getting platforms", __LINE__);

    // Get list of devices
    for (cl_uint p = 0; p < num_platforms; p++)
    {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                             MAX_DEVICES - total_devices, devices + total_devices,
                             &num_devices);
        //checkError(err, "getting device name", __LINE__);
        total_devices += num_devices;
    }

    // Print list of devices
    //printf("\nAvailable OpenCL devices:\n");
    //for (cl_uint d = 0; d < total_devices; d++)
    //{
     //   clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    //    printf("%2d: %s\n", d, name);
    //}
    //printf("\n");

    // Use first device unless OCL_DEVICE environment variable used
    cl_uint device_index = 0;
    char *dev_env = getenv("OCL_DEVICE");
    if (dev_env)
    {
        char *end;
        device_index = strtol(dev_env, &end, 10);
        if (strlen(end))
            die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
    }

    if (device_index >= total_devices)
    {
        fprintf(stderr, "device index set to %d but only %d devices available\n",
                device_index, total_devices);
        exit(1);
    }

    // Print OpenCL device name
    //clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME, MAX_DEVICE_NAME, name,
    //                NULL);
    //printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

    return devices[device_index];
}

char *common_read_file(const char *path, long *length_out)
{
    char message[1024];
    char *buffer;
    FILE *f;
    long length;

    f = fopen(path, "r");
    if (f == NULL)
    {
        sprintf(message, "could not open OpenCL kernel file: %s", OCL_KERNELS_FILE);
        die(message, __LINE__, __FILE__);
    }
    fseek(f, 0, SEEK_END);
    length = ftell(f) + 1;
    fseek(f, 0, SEEK_SET);
    buffer = malloc(length);
    memset(buffer, 0, length);
    fread(buffer, 1, length, f);
    fclose(f);
    *length_out = length;
    return buffer;
}

void OCL_initialise(t_ocl *ocl, const int N)
{

    //char message[1024]; /* message buffer */
    //FILE *fp;           /* file pointer */
    char *ocl_src; /* OpenCL kernel source */
    long ocl_size; /* size of OpenCL kernel source */
    cl_int err;

    // Initialise OpenCL
    // Get an OpenCL device
    ocl->device = selectOpenCLDevice();

    ocl->wgsize = 64;

    // Create OpenCL context
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    //checkError(err, "creating context", __LINE__);

    // Create OpenCL command queue
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    //checkError(err, "creating command queue", __LINE__);

    /*  clCreateProgramWithBinary
    FILE *f;
  char *binary;       // compiled kernel 
  size_t binary_size;
    #define BIN_PATH "kernel.bin"
    binary = common_read_file(BIN_PATH, &binary_size);
  cl_int errcode_ret, binary_status;
  ocl->program = clCreateProgramWithBinary(
        ocl->context, 1, &ocl->device, &binary_size,
        (const unsigned char **)&binary, &binary_status, &errcode_ret);
  clGetProgramInfo(ocl->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
  binary = malloc(binary_size);
  clGetProgramInfo(ocl->program, CL_PROGRAM_BINARIES, binary_size, &binary, NULL);
  f = fopen(BIN_PATH, "w");
  fwrite(binary, binary_size, 1, f);
  fclose(f);

  */

    ocl_src = common_read_file(OCL_KERNELS_FILE, &ocl_size);
    ocl->program = clCreateProgramWithSource(ocl->context, 1,
                                             (const char **)&ocl_src, NULL, &err);
    free(ocl_src);
    //checkError(err, "creating program", __LINE__);

    // Build OpenCL program
    err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t sz;
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0,
                              NULL, &sz);
        char *buildlog = malloc(sz);
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz,
                              buildlog, NULL);
        fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
        free(buildlog);
    }
    //checkError(err, "building program", __LINE__);

    // Create OpenCL kernels
    ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
    //checkError(err, "creating kernel", __LINE__);
    ocl->propagate = clCreateKernel(ocl->program, "propagate", &err);
    //checkError(err, "creating kernel", __LINE__);
    //ocl->rebound = clCreateKernel(ocl->program, "rebound", &err);
    //checkError(err, "creating kernel", __LINE__);
    ocl->rebound_collision = clCreateKernel(ocl->program, "rebound_collision", &err);
    //checkError(err, "creating kernel", __LINE__);
    ocl->av_velocity = clCreateKernel(ocl->program, "av_velocity", &err);
    //checkError(err, "creating kernel", __LINE__);

    // get work group size
    err = clGetKernelWorkGroupInfo(ocl->av_velocity, ocl->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &ocl->av_work_group_size, NULL);
    //ocl->av_work_group_size=64;
    //checkError(err, "Getting kernel work group info",__LINE__);

    // Allocate OpenCL buffers
    ocl->d_obstacles = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
                                      sizeof(cl_int) * N, NULL, &err);
    //checkError(err, "creating buffer a", __LINE__);

    ocl->d_cells = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                  sizeof(t_speed) * N, NULL, &err);
    //checkError(err, "creating buffer b", __LINE__);

    ocl->d_tmp_cells = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                      sizeof(t_speed) * N, NULL, &err);
    //checkError(err, "creating buffer b", __LINE__);

    
    ocl->d_groupsums = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
                                      sizeof(float) * (int)(N / ocl->av_work_group_size) , NULL, &err);
    //checkError(err, "creating buffer b", __LINE__);

    result = malloc(sizeof(float) * (int)(N / ocl->av_work_group_size));

}

float w1,w2;
void OCL_setKernelArgs(const t_param params){

    err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_int), &params.nx);
    //checkError(err, "setting accelerate_flow arg 0", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_int), &params.ny);
    //checkError(err, "setting accelerate_flow arg 1", __LINE__);

    w1 = params.density * params.accel / 9.f;
    w2 = params.density * params.accel / 36.f;

    err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_float), &w1);
    //checkError(err, "setting accelerate_flow arg 2", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_float), &w2);
    //checkError(err, "setting accelerate_flow arg 3", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_mem), &ocl.d_cells);
    //checkError(err, "setting accelerate_flow arg 4", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_mem), &ocl.d_obstacles);
    //checkError(err, "setting accelerate_flow arg 5", __LINE__);

    err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_int), &params.nx);
    //checkError(err, "setting propagate arg 0", __LINE__);
    err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_int), &params.ny);
    //checkError(err, "setting propagate arg 1", __LINE__);
    err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_int), &ll);
    //checkError(err, "setting propagate arg 2", __LINE__);
    err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_mem), &ocl.d_cells);
    //checkError(err, "setting propagate arg 3", __LINE__);
    err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_mem), &ocl.d_tmp_cells);
    //checkError(err, "setting propagate arg 4", __LINE__);

    // err = clSetKernelArg(ocl.rebound, 0, sizeof(cl_int), &params.nx);
    // //checkError(err, "setting rebound arg 0", __LINE__);
    // err = clSetKernelArg(ocl.rebound, 1, sizeof(cl_int), &ll);
    // //checkError(err, "setting rebound arg 1", __LINE__);    
    // err = clSetKernelArg(ocl.rebound, 2, sizeof(cl_mem), &ocl.d_cells);
    // //checkError(err, "setting rebound arg 2", __LINE__);
    // err = clSetKernelArg(ocl.rebound, 3, sizeof(cl_mem), &ocl.d_tmp_cells);
    // //checkError(err, "setting rebound arg 3", __LINE__);
    // err = clSetKernelArg(ocl.rebound, 4, sizeof(cl_mem), &ocl.d_obstacles);
    //checkError(err, "setting propagate arg 4", __LINE__);

    err = clSetKernelArg(ocl.rebound_collision, 0, sizeof(cl_int), &params.nx);
    //checkError(err, "setting rebound_collision arg 0", __LINE__);
    err = clSetKernelArg(ocl.rebound_collision, 1, sizeof(cl_int), &ll);
    //checkError(err, "setting rebound_collision arg 1", __LINE__);
    err = clSetKernelArg(ocl.rebound_collision, 2, sizeof(cl_float), &params.omega);
    //checkError(err, "setting rebound_collision arg 1", __LINE__);    
    err = clSetKernelArg(ocl.rebound_collision, 3, sizeof(cl_mem), &ocl.d_cells);
    //checkError(err, "setting rebound_collision arg 2", __LINE__);
    err = clSetKernelArg(ocl.rebound_collision, 4, sizeof(cl_mem), &ocl.d_tmp_cells);
    //checkError(err, "setting rebound_collision arg 3", __LINE__);
    err = clSetKernelArg(ocl.rebound_collision, 5, sizeof(cl_mem), &ocl.d_obstacles);
    //checkError(err, "setting rebound_collision arg 5", __LINE__);

    err = clSetKernelArg(ocl.av_velocity, 0, sizeof(cl_int), &params.nx);
    //checkError(err, "setting av_velocity arg 0", __LINE__);
    err = clSetKernelArg(ocl.av_velocity, 1, sizeof(cl_int), &ll);
    //checkError(err, "setting av_velocity arg 1", __LINE__);
    err = clSetKernelArg(ocl.av_velocity, 2, sizeof(float)*ocl.av_work_group_size, NULL);
    //checkError(err, "setting av_velocity arg 2", __LINE__); 
    err = clSetKernelArg(ocl.av_velocity, 3, sizeof(cl_mem), &ocl.d_groupsums);
    //checkError(err, "setting av_velocity arg 3", __LINE__); 
    err = clSetKernelArg(ocl.av_velocity, 4, sizeof(cl_mem), &ocl.d_cells);
    //checkError(err, "setting av_velocity arg 4", __LINE__);
    err = clSetKernelArg(ocl.av_velocity, 5, sizeof(cl_mem), &ocl.d_obstacles);
    //checkError(err, "setting av_velocity arg 5", __LINE__);
}
void OCL_finalise(t_ocl ocl)
{
    clReleaseMemObject(ocl.d_obstacles);
    clReleaseMemObject(ocl.d_cells);
    clReleaseMemObject(ocl.d_tmp_cells);
    clReleaseMemObject(ocl.d_groupsums);
    clReleaseKernel(ocl.accelerate_flow);
    clReleaseKernel(ocl.propagate);
    //clReleaseKernel(ocl.rebound);
    clReleaseKernel(ocl.rebound_collision);
    clReleaseKernel(ocl.av_velocity);
    clReleaseProgram(ocl.program);
    clReleaseCommandQueue(ocl.queue);
    clReleaseContext(ocl.context);
    free(result);
}

