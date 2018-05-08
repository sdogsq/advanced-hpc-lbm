/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
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
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#define  MASTER		0

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/

int MRank,MSize;
MPI_Datatype TMPI_T_SPEED;
int ll,rr;
int nonblocked_cells=0;
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */

  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

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

  //------------------------------------------------
  MPI_Init(NULL,NULL);

  /* Get rank */
  MPI_Comm_rank(MPI_COMM_WORLD,&MRank);
  MPI_Comm_size(MPI_COMM_WORLD,&MSize);
  printf("%d %d\n",MRank,MSize);
  MPI_Type_contiguous( NSPEEDS, MPI_FLOAT, &TMPI_T_SPEED );
  MPI_Type_commit(&TMPI_T_SPEED);
  //------------------------------------------------
  
  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels); // MPI changed
  
  MPI_Bcast(&(params.nx),1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&(params.ny),1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&(params.maxIters),1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&(params.reynolds_dim),1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&(params.density),1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&(params.accel),1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&(params.omega),1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  MPI_Bcast(&nonblocked_cells,1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  MPI_Bcast(obstacles,params.nx*params.ny,MPI_INT,MASTER,MPI_COMM_WORLD);
  MPI_Bcast(cells,params.nx*params.ny,TMPI_T_SPEED,MASTER,MPI_COMM_WORLD);

  ll=params.ny/MSize*(MRank);
  rr=(MRank==MSize-1)?params.ny:params.ny/MSize*(MRank+1);

  MPI_Barrier(MPI_COMM_WORLD);
  
  //printf("%d %d %d %d \n",MRank,params.ny,ll,rr);
  //------------------------------------------------
  
  /* iterate for maxIters timesteps */
  if (MRank==MASTER){
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  }

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, obstacles);
    float local_av = av_velocity(params, cells, obstacles);
    float global_av;
    MPI_Reduce(&local_av, &global_av, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    if (MRank==MASTER){av_vels[tt]=global_av;
#ifdef DEBUG
    printf("%d==timestep: %d==\n", MRank,tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }
  }
  // ------------------ 
  if (MRank!=MASTER){
    MPI_Ssend(&(cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, MASTER,3, MPI_COMM_WORLD);
  }else{
    for (int i=1;i<MSize;++i){
      int llx=params.ny/MSize*i;
      int rrx=(i==MSize-1)?params.ny:params.ny/MSize*(i+1);
      MPI_Recv(&(cells[llx*params.nx]),(rrx-llx)*params.nx, TMPI_T_SPEED, i,3, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }
  // ------------------

  if (MRank==MASTER){
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
  MPI_Finalize();
  //-----------------------
  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  accelerate_flow(params, cells, obstacles);
  propagate(params, cells, tmp_cells);
  rebound(params, cells, tmp_cells, obstacles);
  collision(params, cells, tmp_cells, obstacles);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{ 

  if (MRank==MSize-1) {
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;
 
  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  //int ll=params.nx/MSize

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    int idx=ii + jj*params.nx;
    if (!obstacles[idx]
        && (cells[idx].speeds[3] - w1) > 0.f
        && (cells[idx].speeds[6] - w2) > 0.f
        && (cells[idx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[idx].speeds[1] += w1;
      cells[idx].speeds[5] += w2;
      cells[idx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[idx].speeds[3] -= w1;
      cells[idx].speeds[6] -= w2;
      cells[idx].speeds[7] -= w2;
    }
  }
  }
  //MPI_Bcast(&(cells[(params.ny - 2)*params.nx]),params.nx, TMPI_T_SPEED, MASTER, MPI_COMM_WORLD);
  //MPI_Barrier(MPI_COMM_WORLD);
 /* for (int i=0;i<params.nx;++i)
    for (int j=0;i<params.ny;++i){
      printf("%d %f\n",MRank,cells[i+j*params.nx].speeds[1]);
    }*/
  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
  /* loop over _all_ cells */


  int up_rank=(MRank+1)>=MSize?0:MRank+1;
  int down_rank=(MRank-1)<0?(MSize-1):(MRank-1);
  int up_cell=rr>=params.ny?0:rr;
  int down_cell=(ll-1)<0?(params.ny-1):(ll-1);
  MPI_Sendrecv(&(cells[(rr-1)*params.nx]),params.nx,TMPI_T_SPEED,up_rank,0,&(cells[down_cell*params.nx]),
               params.nx,TMPI_T_SPEED,down_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  MPI_Sendrecv(&(cells[ll*params.nx]),params.nx,TMPI_T_SPEED,down_rank,1,&(cells[up_cell*params.nx]),
               params.nx,TMPI_T_SPEED,up_rank,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

  for (int jj = ll; jj < rr; jj++)//for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int idx=ii + jj*params.nx;
      int y_n = (jj + 1)>=params.ny?(jj+1-params.ny):(jj+1); //% params.ny;
      int x_e = (ii + 1)>=params.nx?(ii+1-params.nx):(ii+1); // % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[idx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[idx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[idx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[idx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[idx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[idx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[idx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[idx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[idx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
    }
  }
 /* if (MRank!=MASTER){
    MPI_Ssend(&(tmp_cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, MASTER,0, MPI_COMM_WORLD);
  }else{
    for (int i=1;i<MSize;++i){
      int llx=params.ny/MSize*i;
      int rrx=(i==MSize-1)?params.ny:params.ny/MSize*(i+1);
      MPI_Recv(&(tmp_cells[llx*params.nx]),(rrx-llx)*params.nx, TMPI_T_SPEED, i,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }
  MPI_Bcast(&(tmp_cells[0]),params.ny*params.nx, TMPI_T_SPEED, MASTER, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);*/
  //printf("ABC");
  /*MPI_Allgather(&(tmp_cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, cbuffer2, (rr-ll)*params.nx, TMPI_T_SPEED,MPI_COMM_WORLD);
  t_speed *tmp;
  tmp=tmp_cells;
  tmp_cells=cbuffer2;
  cbuffer2=tmp;*/
  //printf("%d\n",MRank);
  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* loop over the cells in the grid */
  for (int jj = ll; jj < rr; jj++)//for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx=ii + jj*params.nx;
      /* if the cell contains an obstacle */
      if (obstacles[idx])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[idx].speeds[1] = tmp_cells[idx].speeds[3];
        cells[idx].speeds[2] = tmp_cells[idx].speeds[4];
        cells[idx].speeds[3] = tmp_cells[idx].speeds[1];
        cells[idx].speeds[4] = tmp_cells[idx].speeds[2];
        cells[idx].speeds[5] = tmp_cells[idx].speeds[7];
        cells[idx].speeds[6] = tmp_cells[idx].speeds[8];
        cells[idx].speeds[7] = tmp_cells[idx].speeds[5];
        cells[idx].speeds[8] = tmp_cells[idx].speeds[6];
      }
    }
  }
/*
  if (MRank!=MASTER){
    MPI_Ssend(&(cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, MASTER,0, MPI_COMM_WORLD);
  }else{
    for (int i=1;i<MSize;++i){
      int llx=params.ny/MSize*i;
      int rrx=(i==MSize-1)?params.ny:params.ny/MSize*(i+1);
      MPI_Recv(&(cells[llx*params.nx]),(rrx-llx)*params.nx, TMPI_T_SPEED, i,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }
  MPI_Bcast(&(cells[0]),params.ny*params.nx, TMPI_T_SPEED, MASTER, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
*/
  /*MPI_Allgather(&(cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, cbuffer, (rr-ll)*params.nx, TMPI_T_SPEED,MPI_COMM_WORLD);
    t_speed *tmp;
  tmp=cells;
  cells=cbuffer;
  cbuffer=tmp;*/
  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = ll; jj < rr; jj++)//for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx=ii + jj*params.nx;
      /* don't consider occupied cells */
      if (!obstacles[idx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[idx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[idx].speeds[1]
                      + tmp_cells[idx].speeds[5]
                      + tmp_cells[idx].speeds[8]
                      - (tmp_cells[idx].speeds[3]
                         + tmp_cells[idx].speeds[6]
                         + tmp_cells[idx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[idx].speeds[2]
                      + tmp_cells[idx].speeds[5]
                      + tmp_cells[idx].speeds[6]
                      - (tmp_cells[idx].speeds[4]
                         + tmp_cells[idx].speeds[7]
                         + tmp_cells[idx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[idx].speeds[kk] = tmp_cells[idx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[idx].speeds[kk]);
        }
      }
    }
  }
/*  if (MRank!=MASTER){
    MPI_Ssend(&(cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, MASTER,0, MPI_COMM_WORLD);
  }else{
    for (int i=1;i<MSize;++i){
      int llx=params.ny/MSize*i;
      int rrx=(i==MSize-1)?params.ny:params.ny/MSize*(i+1);
      MPI_Recv(&(cells[llx*params.nx]),(rrx-llx)*params.nx, TMPI_T_SPEED, i,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }
  MPI_Bcast(&(cells[0]),params.ny*params.nx, TMPI_T_SPEED, MASTER, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
*/

  /*MPI_Allgather(&(cells[ll*params.nx]),(rr-ll)*params.nx, TMPI_T_SPEED, cbuffer, (rr-ll)*params.nx, TMPI_T_SPEED,MPI_COMM_WORLD);
  t_speed *tmp;
  tmp=cells;
  cells=cbuffer;
  cbuffer=tmp;*/
  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  //int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = ll; jj < rr; jj++)//for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx =ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[idx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[idx].speeds[kk];
        }
        //if (abs(local_density)<=0.0000000000001) printf("!~%d\n",MRank);
        /* x-component of velocity */
        float u_x = (cells[idx].speeds[1]
                      + cells[idx].speeds[5]
                      + cells[idx].speeds[8]
                      - (cells[idx].speeds[3]
                         + cells[idx].speeds[6]
                         + cells[idx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[idx].speeds[2]
                      + cells[idx].speeds[5]
                      + cells[idx].speeds[6]
                      - (cells[idx].speeds[4]
                         + cells[idx].speeds[7]
                         + cells[idx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++tot_cells;
      }
    }
  }
  return tot_u / (float)nonblocked_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

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
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);


   /// ---------------------
  if (MRank!=MASTER){*av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
 return EXIT_SUCCESS;}
   /// ---------------------

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
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
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

// ----------------------------------------------
  for (int jj = 0; jj < params->ny; jj++)
    for (int ii = 0; ii < params->nx; ii++)
      if (!(*obstacles_ptr)[ii + jj*params->nx]) ++nonblocked_cells;
// ----------------------------------------------
  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
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


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  //-------------------
  ll=0,rr=params.nx;
  //-------------------
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int idx = ii + jj*params.nx;
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
        u_x = (cells[idx].speeds[1]
               + cells[idx].speeds[5]
               + cells[idx].speeds[8]
               - (cells[idx].speeds[3]
                  + cells[idx].speeds[6]
                  + cells[idx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[idx].speeds[2]
               + cells[idx].speeds[5]
               + cells[idx].speeds[6]
               - (cells[idx].speeds[4]
                  + cells[idx].speeds[7]
                  + cells[idx].speeds[8]))
              / local_density;
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

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
