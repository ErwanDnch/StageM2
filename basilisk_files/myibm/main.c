#include "tools.h"
#include "RK-substep.h"

#define NB 128
#define MINLEVEL 4
const int MAXLEVEL = 10;

const int Nb = NB;

double s, ds;
double angle = 0.;
double Lp = 0.25;
double Ls = 1.;
double ds;
double mu0 = 1e-6;
int Reynolds = 100;

const double H = 1.72;
const double L = 4*1.72;
const double p0 = 0, pseudo_p0 = 0;

coord B_pointsX[NB], U_tilde[NB], Ud[NB], Bound_F[NB];

scalar grid_points;
scalar a[];

u.n[left]   = dirichlet(0.);
u.n[top]    = dirichlet(0.);
u.n[bottom] = dirichlet(0.);
u.n[right]  = dirichlet(0.);

u.t[left]   = dirichlet(0.);
u.t[top]    = dirichlet(0.);
u.t[bottom] = dirichlet(0.);
u.t[right]  = dirichlet(0.);


u_star.n[top]    = dirichlet(0.);
u_star.n[bottom] = dirichlet(0.);
u_star.n[left]   = dirichlet(0.);
u_star.n[right]  = dirichlet(0.);

u_star.t[top]    = dirichlet(0.);
u_star.t[bottom] = dirichlet(0.);
u_star.t[left]   = dirichlet(0.);
u_star.t[right]  = dirichlet(0.);

p[top] = neumann(0.);
p[bottom] = neumann(0.);
p[right] = neumann(0.);
p[left] = neumann(0.);


event init(t=0){
    foreach(){
        foreach_dimension(){
            u.x[]  = 1.;
        }
        p[] = p0;
        pseudo_p[] = pseudo_p0;
    }
    foreach_face(){
        mu.x[] = mu0;
    }
    for (int k = 0; k<Nb; k++){
        foreach_dimension(){
            Ud[k].x = 0;
        }
    }
}


int main() {
    N = 16;
    L0 = L;
    ds = Nb > 0 ? Ls / Nb : HUGE;
    origin(-H/2, -L/2);

    run();
    return 0;
}




event boundary_points(i++) {
    foreach()
        grid_points[] = 0;
    for (int k = 0; k < Nb; k++) {
        s = k * ds;
        B_pointsX[k].x = (Lp - s) * sin(angle);
        B_pointsX[k].y = (s - Lp) * cos(angle);
        #if dimmension == 3
        B_pointsX[k].z = 0;
        #endif
        //B_pointsX[k][dimension] = Lp - s;
    }
    foreach() {
        for (int k = 0; k < Nb; k++) {
            if (fabs(B_pointsX[k].x - x) < Delta / 2.0 && fabs(B_pointsX[k].y - y) < Delta / 2.0) {
                grid_points[] = 1.;
            }
        }
    }

}

/*
event pic(t=2.){
	static FILE * fp = fopen("grid.ppm", "w");
	output_ppm (grid_points, fp, min = 0, max = 1);

	output_ppm (u.x, "ux.ppm", min = 0, max = 1);
}
*/

event movies (t += 0.024; t <= 2.)
{
  output_ppm (u.x, file = "ux.mp4", min = -10, max = 10, linear = true);
}

event adapt (i++) {
    adapt_wavelet ({u_star,u,f}, (double[]){1e-2,3e-2,3e-2,3e-2}, MAXLEVEL, MINLEVEL);
}

event logfile (i++)
  fprintf (stderr, "%d %g\n", i, t);
