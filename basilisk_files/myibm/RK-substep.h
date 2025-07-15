#include "common.h"
#include "run.h"
#include "poisson.h"


/*
Coefficients from Direct simulation of turbulent flow using finite-difference schemes, Rai and Moan
https://www.sciencedirect.com/science/article/pii/002199919190264L
*/

double coef_alpha[3] = {4./15., 1./15. , 1./6. };
double coef_gamma[3] = {8./15., 5./12. , 3./4. };
double coef_ksi[3]   = {0.    , -17./60, -5./12};

vector u[], u_tilde[], u_temp[], lap_u[], grad_p[], adv_u[], adv_uk2[], mu[], f[], u_star[], grad_pseudo_p[];
scalar p[], pseudo_p[], div_u_star[], lap_pseudo_p[], random_val[], lap_random_val[];

double temp_uk1, temp_uk2;

extern coord B_pointsX[], U_tilde[], Ud[], Bound_F[];
extern const int Nb, MAXLEVEL;
extern double mu0;
double tolerance = 1e-2;




void func_u_tilde(int k){
    if(k != 0){
        foreach_face(){
                foreach_dimension(){
                    u_tilde.x[] = u.x[] + dt/3. * (2. * coef_alpha[k] * mu.x[] * lap_u.x[] -
                        2. * coef_alpha[k] * grad_p.x[] - coef_gamma[k] * adv_u.x[]);
                }
        }
    }else{
        foreach_face(){
                foreach_dimension(){
                    u_tilde.x[] = u.x[] + dt/3. * (2. * coef_alpha[k] * mu.x[] * lap_u.x[] -
                        2. * coef_alpha[k] * grad_p.x[] - coef_gamma[k] * adv_u.x[] - coef_ksi[k] * adv_uk2.x[]);
                    }
                }
    }
}

void func_laplacian_u(){
    foreach(){
        foreach_dimension(){
        lap_u.x[] = (u.x[-1,0] + u.x[1,0] + u.x[0,-1] + u.x[0,1] - 4*u.x[])/sq(Delta);
        }
    }
}

void func_grad_p(){
    foreach(){
        foreach_dimension()
            grad_p.x[] = (p[1, 0] - p[-1, 0])/(2*Delta);
    }
}

void func_advection(){
    foreach(){
        foreach_dimension(){
            foreach_dimension(){
                temp_uk2 = u_temp.x[];
                temp_uk1 = u.x[];
            }
            adv_uk2.x[] = temp_uk2 * (u_temp.x[1,0] - u_temp.x[-1,0])/(2*Delta);
            adv_u.x[] = temp_uk1 * (u.x[1,0] - u.x[-1,0])/(2*Delta);
        }
    }
}


void func_U_tilde(){
    double dirac;
    coord p;
    for(int k = 0; k<Nb; k++){
        foreach_point(B_pointsX[k].x, B_pointsX[k].y){
            foreach_dimension(){
                U_tilde[k].x = 0;
            }
            foreach_neighbor(){
                foreach_dimension(){
                    p.x = x;
                }
                dirac = dirach_delta_func(Delta, p, B_pointsX[k]);
                foreach_dimension(){
                    #if dimension == 2
                    U_tilde[k].x += u_tilde.x[] * dirac * pow(Delta, 2);
                    #elif dimension == 3
                    U_tilde[k].x += u_tilde.x[] * dirac * pow(Delta, 3);
                    #endif
                }
            }
        }
    }
}


void func_bound_F(){
    for(int k = 0; k < Nb; k++){
        foreach_dimension(){
            Bound_F[k].x = (Ud[k].x - U_tilde[k].x)/dt*3;
        }
    }
}


void func_f(){
    double dirac;
    coord p;
    for(int k = 0; k<Nb; k++){
        foreach_point(B_pointsX[k].x, B_pointsX[k].y){
            foreach_dimension(){
                U_tilde[k].x = 0;
            }
            foreach_neighbor(){
                foreach_dimension(){
                    p.x = x;
                }
                dirac = dirach_delta_func(Delta, p, B_pointsX[k]);
                foreach_dimension(){
                    #if dimension == 2
                    f.x[] += Bound_F[k].x * dirac * sq(Delta);
                    #elif dimension == 3
                    f.x[] += Bound_F[k].x * dirac * pow(Delta, 3);
                    #endif
                }
            }
        }
    }
}


void func_u_star(int k){
    const face vector alpha = {{1}};
    scalar lambda[];
    vector b[];
    foreach(){
        lambda[] = -1/(coef_alpha[k] * mu.x[] * dt / 3);
        foreach_dimension(){
            b.x[] = -1. / (mu.x[] * coef_alpha[k]) * (u_tilde.x[]/dt * 3 + f.x[]) + lap_u.x[];
        }
    }
    foreach_dimension()
        poisson(u_star.x, b.x, alpha, lambda, tolerance);
}

void func_pseudo_p(int k){
    scalar b[];
    foreach(){
        //fprintf(stderr, "%g %g %g %g %d %g\n", u_star.x[], div_u_star[], Delta, x, k, t);
        b[] = div_u_star[] / (2*coef_alpha[k]*dt/3);
    }
    poisson(pseudo_p, b, tolerance = tolerance);
}

void init(){
    foreach_face(){
        mu.x[] = mu0;
    }
}


void update_u(int k){
    gradient(pseudo_p, grad_pseudo_p);
    foreach(){
        foreach_dimension(){
            u.x[] = u_star.x[] - 2*coef_alpha[k] * dt/3 * grad_pseudo_p.x[];
        }
        fprintf(stderr, "ux: %g   uy: %g   pseudo_p: %g   p: %g   k: %d   t: %g\n", u.x[], u.y[], pseudo_p[], p[], k, t);
    }
}


void update_p(int k){
    scalar_laplacian(pseudo_p, lap_pseudo_p);
    foreach(){
        //fprintf(stderr, "%g %g %g %g %g %g\n",pseudo_p[], coef_alpha[k], mu.x[], lap_pseudo_p[], x, dt);
        p[] += pseudo_p[] - coef_alpha[k] * dt/3 * mu.x[] * lap_pseudo_p[];
    }
}

void test_lap(){
    foreach(){
        random_val[]     = 100;
    }

    scalar_laplacian(random_val, lap_random_val);
    foreach(){
        fprintf(stderr, "%g %g\n", random_val[], lap_random_val[]);
    }
}


/*
event set_dt(i++){
    double dx = L0/N/pow(2, MAXLEVEL);
    double umax = 0.;
    foreach(){
        foreach_dimension(){
            if(fabs(u.x[]) >= umax){umax = fabs(u.x[]);}
        }
    }
    dt = CFL * dx / umax;
}
*/



event RK3(i++){
    u_temp = u;


    for (int k = 0; k < 3; k++){
        fprintf (stderr, "substep: %d\n", k);
        init();
        func_laplacian_u();
        func_advection();
        func_u_tilde(k);
        func_U_tilde();
        func_bound_F();
        func_u_star(k);
        divergence(u_star, div_u_star);
        //test_lap();
        func_pseudo_p(k);
        update_u(k);
        update_p(k);
    }
}
