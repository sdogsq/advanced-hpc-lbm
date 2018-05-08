#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
typedef struct //__attribute__ ((aligned))
{
    float speeds[NSPEEDS];
} t_speed;

__kernel void accelerate_flow(const int nx,
                              const int ny,
                              const float w1,
                              const float w2,
                              __global t_speed *cells,
                              const __global int *obstacles)
{
    int lx = get_global_id(0);
    int idx = lx + (ny - 2) * nx;
    if (!obstacles[idx] && (cells[idx].speeds[3] - w1) > 0.f && (cells[idx].speeds[6] - w2) > 0.f && (cells[idx].speeds[7] - w2) > 0.f)
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

__kernel void propagate(const int nx,
                        const int ny,
                        const int ll,
                        __global t_speed *cells,
                        __global t_speed *tmp_cells)
{
    int lx = get_global_id(0);
    int ly = get_global_id(1) + ll;

    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int idx = lx + ly * nx;
    int y_n = (ly + 1) >= ny ? (ly + 1 - ny) : (ly + 1); //% ny;
    int x_e = (lx + 1) >= nx ? (lx + 1 - nx) : (lx + 1); // % nx;
    int y_s = (ly == 0) ? (ny - 1) : (ly - 1);
    int x_w = (lx == 0) ? (nx - 1) : (lx - 1);
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    t_speed tmp;
    tmp.speeds[0] = cells[lx + ly * nx].speeds[0];   /* central cell, no movement */
    tmp.speeds[1] = cells[x_w + ly * nx].speeds[1];  /* east */
    tmp.speeds[2] = cells[lx + y_s * nx].speeds[2];  /* north */
    tmp.speeds[3] = cells[x_e + ly * nx].speeds[3];  /* west */
    tmp.speeds[4] = cells[lx + y_n * nx].speeds[4];  /* south */
    tmp.speeds[5] = cells[x_w + y_s * nx].speeds[5]; /* north-east */
    tmp.speeds[6] = cells[x_e + y_s * nx].speeds[6]; /* north-west */
    tmp.speeds[7] = cells[x_e + y_n * nx].speeds[7]; /* south-west */
    tmp.speeds[8] = cells[x_w + y_n * nx].speeds[8]; /* south-east */
    tmp_cells[idx]=tmp;
}

// __kernel void rebound(const int nx,
//                       const int ll,
//                       __global t_speed *cells,
//                       const __global t_speed *tmp_cells,
//                       const __global int *obstacles)
// {
//     int lx = get_global_id(0);
//     int ly = get_global_id(1) + ll;
//     /* loop over the cells in the grid */
//     int idx = lx + ly * nx;
//     /* if the cell contains an obstacle */
//     if (obstacles[idx])
//     {
//         t_speed tmp,tmpb;
//         tmpb=tmp_cells[idx];
//         /* called after propagate, so taking values from scratch space
//         ** mirroring, and writing into main grid */
//         tmp.speeds[1] = tmpb.speeds[3];
//         tmp.speeds[2] = tmpb.speeds[4];
//         tmp.speeds[3] = tmpb.speeds[1];
//         tmp.speeds[4] = tmpb.speeds[2];
//         tmp.speeds[5] = tmpb.speeds[7];
//         tmp.speeds[6] = tmpb.speeds[8];
//         tmp.speeds[7] = tmpb.speeds[5];
//         tmp.speeds[8] = tmpb.speeds[6];
//         cells[idx]=tmp;
//     }
// }

__kernel void rebound_collision(const int nx,
                        const int ll,
                        const float omega,
                        __global t_speed *cells,
                        const __global t_speed *tmp_cells,
                        const __global int *obstacles)
{
    const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;   /* weighting factor */
    const float w1 = 1.f / 9.f;   /* weighting factor */
    const float w2 = 1.f / 36.f;  /* weighting factor */

    /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */

    int lx = get_global_id(0);
    int ly = get_global_id(1) + ll;
    int idx = lx + ly * nx;

    if (obstacles[idx])
    {
        t_speed tmp,tmpb;
        tmpb=tmp_cells[idx];
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp.speeds[1] = tmpb.speeds[3];
        tmp.speeds[2] = tmpb.speeds[4];
        tmp.speeds[3] = tmpb.speeds[1];
        tmp.speeds[4] = tmpb.speeds[2];
        tmp.speeds[5] = tmpb.speeds[7];
        tmp.speeds[6] = tmpb.speeds[8];
        tmp.speeds[7] = tmpb.speeds[5];
        tmp.speeds[8] = tmpb.speeds[6];
        cells[idx]=tmp;
    }
    /* don't consider occupied cells */
    if (!obstacles[idx])
    {
        t_speed tmp=tmp_cells[idx]; //cache
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
            local_density += tmp.speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp.speeds[1] + tmp.speeds[5] + tmp.speeds[8] - (tmp.speeds[3] + tmp.speeds[6] + tmp.speeds[7])) / local_density;
        /* compute y velocity component */
        float u_y = (tmp.speeds[2] + tmp.speeds[5] + tmp.speeds[6] - (tmp.speeds[4] + tmp.speeds[7] + tmp.speeds[8])) / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] = u_x;        /* east */
        u[2] = u_y;        /* north */
        u[3] = -u_x;       /* west */
        u[4] = -u_y;       /* south */
        u[5] = u_x + u_y;  /* north-east */
        u[6] = -u_x + u_y; /* north-west */
        u[7] = -u_x - u_y; /* south-west */
        u[8] = u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        float a = 1 / c_sq;
        float b = 1 / (2.f * c_sq * c_sq);
        float c = u_sq / ( 2.f * c_sq);
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.f - c);
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] * a + (u[1] * u[1]) * b - c);
        d_equ[2] = w1 * local_density * (1.f + u[2] * a + (u[2] * u[2]) * b - c);
        d_equ[3] = w1 * local_density * (1.f + u[3] * a + (u[3] * u[3]) * b - c);
        d_equ[4] = w1 * local_density * (1.f + u[4] * a + (u[4] * u[4]) * b - c);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] * a + (u[5] * u[5]) * b - c);
        d_equ[6] = w2 * local_density * (1.f + u[6] * a + (u[6] * u[6]) * b - c);
        d_equ[7] = w2 * local_density * (1.f + u[7] * a + (u[7] * u[7]) * b - c);
        d_equ[8] = w2 * local_density * (1.f + u[8] * a + (u[8] * u[8]) * b - c);

        /* relaxation step */
        t_speed tmp2; // cache
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
            tmp2.speeds[kk] = tmp.speeds[kk] + omega * (d_equ[kk] - tmp.speeds[kk]);
        }
        cells[idx]=tmp2;
    }
}

__kernel void av_velocity(const int nx,
                          const int ll,
                          __local float *localsums,
                          __global float *groupsums,
                          const __global t_speed *cells,
                          const __global int *obstacles)
{

    /* initialise */
    int localID = get_local_id(0);
    int globalID = get_global_id(0);
    int groupID = get_group_id(0);
    int localSize = get_local_size(0);
    int idx = globalID + ll * nx;

    /* loop over all non-blocked cells */
    /* ignore occupied cells */
    float tot_u = 0.f; 
    if (!obstacles[idx])
    {
        /* local density total */
        float local_density = 0.f;
        t_speed tmp=cells[idx]; // cache

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
            local_density += tmp.speeds[kk];
        }
        /* x-component of velocity */
        float u_x = (tmp.speeds[1] + tmp.speeds[5] + tmp.speeds[8] - (tmp.speeds[3] + tmp.speeds[6] + tmp.speeds[7])) / local_density;
        /* compute y velocity component */
        float u_y = (tmp.speeds[2] + tmp.speeds[5] + tmp.speeds[6] - (tmp.speeds[4] + tmp.speeds[7] + tmp.speeds[8])) / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u = sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
    }
    localsums[localID] = tot_u;

    for (int offset = (localSize >> 1); offset > 0; offset >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE); // wait for all other work-items to finish previous iteration.
        if (localID < offset)
        {
            localsums[localID] += localsums[localID + offset];
        }
    }

    if (localID == 0)
    { // the root of the reduction subtree
        groupsums[groupID] = localsums[0];
    }
}