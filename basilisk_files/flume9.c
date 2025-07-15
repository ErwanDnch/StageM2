#include "embed.h"
#include "navier-stokes/centered.h"
#include "tracer.h"
#include "vtk.h"
#include "output_fields/vtu/output_vtu.h"

#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>


scalar f[];
scalar * tracers = {f};
double Reynolds = 2000.;
int maxlevel = 9;
face vector muv[];
char dir_name[90];
char dir_vids[100];
char dir_dumps[200];
char temp_file_name[150];
char file_name[200];
char name[250];
float t_dump;

coord Fp, Fmu;

FILE * fp;

#define H 1.72
#define TMAX 110
#define TGETDATA 10

int rank;
clock_t start_cpu;
struct timeval start_t;

int main()
{
  L0 = H ;         // The domain's lenght is 4 times its wdth
  origin (-0.5, -L0/2.);
  N = 2 << maxlevel;
  mu = muv;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  start_cpu = clock();
  gettimeofday(&start_t, NULL);
  
  TOLERANCE = 1e-6;
  run();
}

double D = 0.1, U0 = 0.24;

event properties (i++)
{
  foreach_face()
    muv.x[] = fm.x[]*D*U0/Reynolds;
}

u.n[left]  = dirichlet(U0);
p[left]    = neumann(0.);
pf[left]   = neumann(0.);
f[left]    = dirichlet(y < 0);

u.n[right] = neumann(0.);
p[right]   = dirichlet(0.);
pf[right]  = dirichlet(0.);

// The walls are no slip
u.n[embed] = dirichlet(0.);
u.t[embed] = dirichlet(0.);


u.n[bottom] = dirichlet(0.);
u.t[bottom] = dirichlet(0.);

u.n[top] = dirichlet(0.);
u.t[top] = dirichlet(0.);

event init (t = 0)
{
  solid (cs, fs, sqrt(sq(x) + sq(y)) - D/2.);

  /**
  We set the initial velocity field. */
  
  if (restore ("restart")){
    fprintf (stderr, "RESTARTING FROM %g s\n", t_dump);
    sprintf(dir_name, "results_U%g_D%g_N%d_LEVEL%d_fromT%g", U0, D, N, maxlevel,t);
  }else{
    sprintf(dir_name, "results_U%g_D%g_N%d_LEVEL%d", U0, D, N, maxlevel);
    double mu_scalar = D*U0/Reynolds; 
    foreach(){
      u.x[] = cs[] ? U0 : 0.;
      f[]   = cs[] ? (y<0) : 0;
      }  
    }
  mkdir(dir_name, 0777);
  sprintf(dir_dumps, "%s/Dumps", dir_name);
  mkdir(dir_dumps,0777);
}


// mgp = Poisson, mgu = Viscosité
event logfile (i++){
  clock_t now_cpu = clock();
  struct timeval now_t;
  gettimeofday(&now_t, NULL);

  double cpu_time_local = (double)(now_cpu - start_cpu) / CLOCKS_PER_SEC;
  double elapsed_t = (now_t.tv_sec - start_t.tv_sec) + (now_t.tv_usec - start_t.tv_usec)/1e6;

  #if _MPI
    double cpu_time_total;
    MPI_Reduce(&cpu_time_local, &cpu_time_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  #endif
  fprintf (stderr, "ite: %d | Sim T: %.4f | Poisson ite: %d | Vis ite: %d | Elapsed: %.2f | Total CPU TIME: %.2f\n", i, t, mgp.i, mgu.i, elapsed_t, cpu_time_local);
}

event movies (t += 0.05)
{
  scalar omega[], m[];
  vorticity (u, omega);

  if (t>=TGETDATA){
    sprintf(file_name,"%s/results%g", dir_name, t*40);
    output_vtu({f,p, omega}, {u}, file_name);
  }

}


event out_forces (i++; t<=20) {
    embed_force(p, u, mu, &Fp, &Fmu);
    if (rank == 0) {
        char path[130];
        sprintf(path, "./%s/Forces", dir_name);
        fp = (t == 0) ? fopen(path, "w") : fopen(path, "a");
        if (fp) {
            fprintf(fp, "%g %g %g %g %g\n", Fp.x, Fp.y, Fmu.x, Fmu.y, t);
            fclose(fp);
        } else {
            fprintf(stderr, "Error opening file %s\n", path);
        }
    }
}


event adapt (i++) {
    adapt_wavelet ({cs,u, p, f}, (double[]){1e-3,1e-3,1e-3, 1e-3, 1e-2}, maxlevel, 4);
}

event save_dumps(t+=5){
    snprintf (name, sizeof(name), "%s/dump-U%g-D%g-N%d-LEVEL%d-T%.2f",dir_dumps,U0, D, N, maxlevel, t);
    dump (file = name);
    t_dump = t;
}


event end(t=TMAX){
    sprintf (name, "%s/dump-U%g-D%g-N%d-LEVEL%d-T%d",dir_dumps,U0, D, N, maxlevel, TMAX);
    dump (file = name);
    t_dump = t;
}
